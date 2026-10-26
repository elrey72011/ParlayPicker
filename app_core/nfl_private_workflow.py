"""Guided owner workflow using the released NFL inputs, transports and caller."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
import hmac

from app_core import nfl_workflow_inputs as inputs, nfl_workflow_acquisition as acquisition
from app_core import nfl_scenario_research as scenario, nfl_scoring_profile as profile
from app_core.nfl_workflow_store import Store, require, WorkflowError, digest, utc

MAX_GAMES = 4
_ISSUER = object()


@dataclass(frozen=True)
class OwnerAccess:
    owner: str
    proof: object


def authenticate(supplied, configured):
    require(isinstance(configured,str) and len(configured)>=16 and isinstance(supplied,str)
        and hmac.compare_digest(supplied.encode(),configured.encode()),'NFL_WORKFLOW_OWNER_REQUIRED')
    return OwnerAccess('Robert Velarde',_ISSUER)


def production_service(setting, access):
    """Lazy deployment boundary; never called on inactive/locked UI paths."""
    dsn=setting('PARLAYPICKER_NFL_DATABASE_URL','')
    key=setting('PARLAYPICKER_NFL_STORE_KEY','')
    require(dsn and isinstance(key,str) and len(key.encode())>=32,'PRIVATE_STORE_CONFIGURATION_REQUIRED')
    store=Store.postgres(dsn,key.encode(),owner=access.owner)
    return Workflow(store,access,
        allow_acquisition=setting('PARLAYPICKER_NFL_ACQUISITION_ENABLED','false')=='true',
        allow_inference=setting('PARLAYPICKER_NFL_INFERENCE_ENABLED','false')=='true',
        credential=lambda:setting('ODDS_API_KEY',''))


class Workflow:
    def __init__(self,store,access,*,clock=utc,allow_acquisition=False,allow_inference=False,credential=None,score_transport=None,market_transport=None):
        require(type(access) is OwnerAccess and access.proof is _ISSUER and access.owner==store.owner,'NFL_WORKFLOW_OWNER_REQUIRED')
        require(store.synthetic or (score_transport is None and market_transport is None),'NFL_WORKFLOW_SYNTHETIC_INJECTION_FORBIDDEN')
        self.store,self.access,self.clock=store,access,clock
        self.allow_acquisition,self.allow_inference=allow_acquisition,allow_inference
        self.credential=credential
        self.score_transport,self.market_transport=score_transport,market_transport

    def verify(self):
        return self.store.verify()

    def _operation(self,kind,semantic,payload,*,predecessor=None):
        require(kind in {'scores','market'},'NFL_WORKFLOW_OPERATION_UNSUPPORTED')
        method=acquisition.method_binding()
        semantic=dict(semantic,method_sha256=digest(method))
        if predecessor:
            prior=self.store.get(predecessor,kind='operation')
            require(prior['payload']['semantic']==semantic,'NFL_WORKFLOW_SUCCESSOR_TARGET_CONFLICT')
            require(self.store.spent('operation:'+predecessor) or inputs.instant(self.clock())>=inputs.instant(prior['payload']['window']['not_after'])-timedelta(seconds=60), 'NFL_WORKFLOW_PREDECESSOR_STILL_AVAILABLE')
        identity='operation:'+digest(dict(kind=kind,semantic=semantic,predecessor=predecessor))
        at=inputs.instant(self.clock());end=at+timedelta(minutes=20)
        events=payload.get('events',[])
        if events:
            end=min(end,min(inputs.instant(e['start']) for e in events)-timedelta(minutes=1))
        require(end>at+timedelta(minutes=1),'NFL_WORKFLOW_KICKOFF_TOO_CLOSE')
        data=dict(payload,kind=kind,semantic=semantic,predecessor=predecessor,
            method=method,
            window=dict(not_before=at.isoformat(),not_after=end.isoformat()),
            model_sha256=digest(scenario.model_binding()),reader_sha256=digest(scenario.reader_binding()))
        return self.store.put('operation',data,identity=identity,first=True)

    def source_operation(self,*,predecessor=None):
        semantic=dict(source=profile.SOURCE,limits=acquisition.SCORING_LIMITS)
        return self._operation('scores',semantic,dict(request=deepcopy(acquisition.scores.PRIMARY),
            redirect=deepcopy(acquisition.scores.REDIRECT),maximum_http_requests=2,maximum_odds_credits=0,
            maximum_extra_spending=0,events=[]),predecessor=predecessor)

    def capture_source(self,operation_id,*,private_retention,private_research,permission_end):
        require(self.allow_acquisition,'NFL_WORKFLOW_ACQUISITION_NOT_ENABLED')
        require(private_retention is True and private_research is True,'NFL_WORKFLOW_SOURCE_REVIEW_REQUIRED')
        operation=self.store.get(operation_id,kind='operation');require(operation['payload']['kind']=='scores','NFL_WORKFLOW_OPERATION_UNSUPPORTED')
        at=self.clock();end=inputs.instant(permission_end)
        require(end>inputs.instant(operation['payload']['window']['not_after']),'NFL_WORKFLOW_SOURCE_PERMISSION_EXPIRED')
        permission=dict(reviewer='Robert Velarde',reviewed_at=at,effective_from=at,effective_until=permission_end,
            source_id=profile.SOURCE_ID,endpoint=profile.ENDPOINT,private_retention=True,private_research_use=True,
            credential_exclusion=True,basis='Actual owner authorized the displayed bounded public NFL scoring capture and private retention/research through the shown permission end. '+profile.SOURCE['attribution']+' '+profile.SOURCE['license'])
        authority=dict(reviewed_at=at,reviewer='Robert Velarde',scope_sha256=digest(operation['payload']),
            maximum_http_requests=2,maximum_odds_credits=0,maximum_extra_spending=0,
            private_retention=private_retention,private_research_use=private_research)
        return acquisition.capture_scores(self.store,operation,authority,permission,transport=self.score_transport,wall=self.clock)

    def source(self):
        records=self.store.list('source',limit=1)
        require(bool(records),'NFL_WORKFLOW_SCORING_REQUIRED')
        return records[0]

    def games(self):
        return inputs.eligible_games(self.source()['payload'],self.clock())

    def prepare(self,event_ids,*,predecessors=None):
        require(isinstance(event_ids,list) and 1<=len(event_ids)<=MAX_GAMES and len(set(event_ids))==len(event_ids),'NFL_WORKFLOW_GAME_SELECTION_REQUIRED')
        source=self.source();catalog=inputs.eligible_games(source['payload'],self.clock())
        rows={row['event']['canonical_event_id']:row for row in catalog['games']}
        require(all(identity in rows for identity in event_ids),'NFL_WORKFLOW_EVENT_NOT_ELIGIBLE')
        prepared=[]
        for identity in event_ids:
            facts=inputs.prepare_game(source['payload'],rows[identity],self.clock())
            saved=self.store.put('prepared-game',facts)
            request=acquisition.request_for(facts)
            semantic=dict(event=facts['selected']['event'],source_sha256=source['payload']['object']['body_sha256'],
                permission_sha256=digest(source['payload']['permission']),request=request,model=facts['model'],reader=facts['reader'])
            operation=self._operation('market',semantic,dict(prepared_id=saved['id'],request=request,
                events=[facts['selected']['event']],maximum_http_requests=1,maximum_odds_credits=1,maximum_extra_spending=0),
                predecessor=(predecessors or {}).get(identity))
            # Repeated tabs adopt the first durable preparation/clock, not a new window.
            facts=self.store.get(operation['payload']['prepared_id'],kind='prepared-game')['payload']
            prepared.append(dict(operation=operation,prepared=facts))
        return prepared

    def capture_quotes(self,operation_ids,*,remaining_credits,reset_at,private_use,no_overage):
        require(self.allow_acquisition,'NFL_WORKFLOW_ACQUISITION_NOT_ENABLED')
        require(isinstance(operation_ids,list) and 1<=len(operation_ids)<=MAX_GAMES and len(set(operation_ids))==len(operation_ids),'NFL_WORKFLOW_GAME_SELECTION_REQUIRED')
        require(type(remaining_credits) is int and remaining_credits>=len(operation_ids) and private_use is True and no_overage is True,'NFL_WORKFLOW_ACCOUNT_REVIEW_REQUIRED')
        operations=[self.store.get(key,kind='operation') for key in operation_ids]
        require(len({op['payload']['events'][0]['canonical_event_id'] for op in operations})==len(operations),'NFL_WORKFLOW_DUPLICATE_EVENT')
        at=self.clock();batch_scope=digest(operation_ids)
        for op in operations:
            require(op['payload']['kind']=='market','NFL_WORKFLOW_OPERATION_UNSUPPORTED')
            acquisition.check_scope(op,at)
            require(inputs.instant(reset_at)>inputs.instant(op['payload']['window']['not_after']),'NFL_WORKFLOW_ACCOUNT_EXPIRED')
            require(not self.store.spent('operation:'+op['id']),'PRIVATE_OPERATION_ALREADY_SPENT_NO_RETRY')
        ready=[]
        for op in operations:
            facts=self.store.get(op['payload']['prepared_id'],kind='prepared-game')['payload']
            # Recheck history, source permission and model before the short-lived quote.
            current=inputs.prepare_game(facts['source'],facts['selected'],self.clock())
            require(current['features']==facts['features'] and current['model']==facts['model'] and current['reader']==facts['reader'],'NFL_WORKFLOW_INPUTS_CHANGED')
            ready.append((op,facts))
        captured=[]
        for op,facts in ready:
            authority=dict(reviewed_at=at,reviewer='Robert Velarde',scope_sha256=digest(op['payload']),
                batch_scope_sha256=batch_scope,batch_maximum_credits=len(operations),remaining_included_credit_floor=remaining_credits,
                reset_at=reset_at,private_use=True,no_paid_overage=True,maximum_http_requests=1,
                maximum_odds_credits=1,maximum_extra_spending=0)
            result=acquisition.capture_market(self.store,op,authority,facts,credential=self.credential,
                transport=self.market_transport,wall=self.clock)
            captured.append(result)
        return captured

    def stage(self,operation_ids):
        require(isinstance(operation_ids,list) and 1<=len(operation_ids)<=MAX_GAMES,'NFL_WORKFLOW_GAME_SELECTION_REQUIRED')
        staged=[]
        for key in operation_ids:
            op=self.store.get(key,kind='operation')
            prepared=self.store.get(op['payload']['prepared_id'],kind='prepared-game')['payload']
            market=self.store.get('market:'+op['id'],kind='market')
            # Deterministic key keeps the original packet availability clock across reruns.
            identity='pair:'+digest(dict(operation=op['id'],market=market['payload']['object']['body_sha256'],model=prepared['model'],reader=prepared['reader']))
            packet_list=inputs.packets(prepared,market['payload']['object'],self.clock())
            pair=self.store.put('scenario-pair',dict(operation_id=op['id'],prepared=prepared,
                market=market['payload'],packets=packet_list),identity=identity,first=True)
            staged.append(pair)
        return staged

    def calculate(self,pair_ids,findings,*,attestation,limitations,material_context='Not supplied; no explicit player adjustment.'):
        require(self.allow_inference,'NFL_WORKFLOW_INFERENCE_NOT_ENABLED')
        require(attestation is True and limitations is True,'NFL_WORKFLOW_OWNER_REVIEW_REQUIRED')
        require(isinstance(pair_ids,list) and 1<=len(pair_ids)<=MAX_GAMES and len(set(pair_ids))==len(pair_ids),'NFL_WORKFLOW_GAME_SELECTION_REQUIRED')
        require(set(findings)==set(inputs.FINDINGS) and all(v is True for v in findings.values()),'NFL_WORKFLOW_OWNER_FINDINGS_REQUIRED')
        pairs=[self.store.get(key,kind='scenario-pair') for key in pair_ids]
        require(len({p['payload']['prepared']['selected']['event']['canonical_event_id'] for p in pairs})==len(pairs),'NFL_WORKFLOW_DUPLICATE_EVENT')
        reviewed_at=self.clock();approved=[]
        for pair in pairs:
            value=pair['payload'];inputs.observed_pair(value['market']['object'],value['prepared'],reviewed_at)
            operation=self.store.get(value['operation_id'],kind='operation')
            require(operation['payload']['method']==acquisition.method_binding(),'NFL_WORKFLOW_METHOD_CHANGED')
            require(value['prepared']['model']==scenario.model_binding() and value['prepared']['reader']==scenario.reader_binding(),'NFL_WORKFLOW_INPUTS_CHANGED')
            for packet in value['packets']:
                subject=scenario.subject(packet['payload'])
                require(not self.store.spent('native:'+subject),'PRIVATE_OPERATION_ALREADY_SPENT_NO_RETRY')
                basis='Actual owner reviewed the displayed original event, half-point assumptions, permissions/clocks/custody, history/features and fixed-model limitations for '+packet['payload']['event']['canonical_event_id']+'. Evidence source '+packet['payload']['scoring']['body_sha256']+'. Market context '+value['market']['object']['body_sha256']+'. Review is based on displayed reproducible checks, not independent technical audit; no explicit QB/injury adjustment.'
                packet=inputs.reviewed(packet,findings,reviewed_at,basis=basis)
                approved.append((pair,packet))
        # Durable real answer is recorded before any numerical call. No inferred approval.
        review=self.store.put('inference-review',dict(at=reviewed_at,pair_ids=pair_ids,findings=findings,
            attestation=inputs.ATTESTATION,limitations_accepted=True,material_context=material_context,
            scope_sha256=digest([scenario.subject(packet['payload']) for _,packet in approved]),maximum_native_calls=len(approved),provider_requests=0))
        receipts=[]
        for pair,packet in approved:
            subject=scenario.subject(packet['payload'])
            inputs.observed_pair(pair['payload']['market']['object'],pair['payload']['prepared'],self.clock())
            scenario.checked(packet,self.clock())
            self.store.claim('native:'+subject,dict(actual_review_id=review['id'],subject_sha256=subject,maximum_native_calls=1,at=self.clock()))
            try:
                inference_at=self.clock()
                inputs.observed_pair(pair['payload']['market']['object'],pair['payload']['prepared'],inference_at)
                result=scenario.compute(packet,at=inference_at)  # Unchanged native formula/caller.
                receipt=self.store.save_native(result)
                self.store.put('native-outcome',dict(status='COMPLETE',receipt=receipt,at=self.clock()),identity='native-outcome:'+subject)
                receipts.append(receipt)
            except Exception as error:
                self.store.put('native-outcome',dict(status='FAILED_OR_UNSAVED_NO_RETRY',reason=acquisition.safe_code(error),at=self.clock()),identity='native-outcome:'+subject)
                raise
        # Only compare after every original has been durably authenticated/read back.
        results=[self.store.read_native(receipt) for receipt in receipts]
        self.store.put('completed-batch',dict(review_id=review['id'],receipts=receipts,at=self.clock()))
        return results

    def saved(self, *, page=1):
        require(type(page) is int and 1<=page<=1_000_000,'PRIVATE_RECORD_LIMIT')
        return self.store.list('saved-index',offset=100*(page-1))

    def view(self,receipt):
        return self.store.read_native(receipt)
