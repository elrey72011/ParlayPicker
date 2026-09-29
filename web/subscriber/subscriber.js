(function(root){
  'use strict';

  const API='/api/v1';
  const POLL_MS=60000;

  class ApiError extends Error{
    constructor(status,body){
      super(`HTTP ${status}`);
      this.name='ApiError';
      this.status=status;
      this.body=body;
    }
  }

  function createSubscriberApp(environment={}){
    const doc=environment.document||root.document;
    const win=environment.window||root;
    const fetchImpl=environment.fetch||root.fetch.bind(root);
    const location=environment.location||root.location;
    const timers=environment.timers||root;
    const state={
      authenticated:false,
      account:null,
      offer:null,
      results:[],
      latestCurrentRequest:0,
      acceptedPublishedAt:0,
      acceptedRevision:null,
      currentExpiryTimer:null,
      pollTimer:null,
      cancelArmed:false,
      checkoutReturnHandled:false,
      bound:false,
    };
    const element=id=>doc.getElementById(id);
    const safe=value=>value===null||value===undefined||value===''?'Unavailable':String(value);
    const text=(id,value)=>{const node=element(id);if(node)node.textContent=value;};
    const setNotice=(id,value,kind='')=>{const node=element(id);if(!node)return;node.textContent=value;if(kind)node.dataset.state=kind;else delete node.dataset.state;};
    const formatDate=value=>{
      if(!value)return 'Unavailable';
      const parsed=new Date(value);
      return Number.isFinite(parsed.getTime())?parsed.toLocaleString():safe(value);
    };
    const reasonCodes=body=>{
      const detail=body&&body.detail;
      const codes=(detail&&detail.reason_codes)||body&&body.reason_codes;
      return Array.isArray(codes)?codes:[];
    };
    const cookie=name=>{
      const escaped=name.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
      const match=doc.cookie.match(new RegExp(`(?:^|; )${escaped}=([^;]*)`));
      return match?decodeURIComponent(match[1]):'';
    };
    const requestKey=()=>root.crypto&&typeof root.crypto.randomUUID==='function'?root.crypto.randomUUID():`${Date.now()}-${Math.random()}`;

    async function request(path,{method='GET',json=null,idempotency=false}={}){
      const mutation=method!=='GET';
      const headers={Accept:'application/json'};
      if(mutation){
        const csrf=cookie('pp_csrf');
        if(csrf)headers['X-CSRF-Token']=csrf;
        if(json!==null)headers['Content-Type']='application/json';
        if(idempotency)headers['Idempotency-Key']=requestKey();
      }
      const url=path.startsWith('/auth/')?path:API+path;
      let response;
      try{
        response=await fetchImpl(url,{credentials:'include',method,headers,...(json===null?{}:{body:JSON.stringify(json)})});
      }catch(error){
        throw new ApiError(0,{detail:'network unavailable',cause:String(error)});
      }
      const raw=response.status===204?'':await response.text();
      let body=null;
      if(raw){try{body=JSON.parse(raw);}catch{body={detail:'invalid server response'};}}
      if(!response.ok)throw new ApiError(response.status,body);
      return body;
    }

    function appendFacts(container,facts){
      const dl=doc.createElement('dl');
      dl.className='facts';
      for(const [label,value] of facts){
        const dt=doc.createElement('dt');dt.textContent=label;
        const dd=doc.createElement('dd');dd.textContent=safe(value);
        dl.append(dt,dd);
      }
      container.append(dl);
    }

    function renderPick(item,actionable=true){
      const article=doc.createElement('article');
      article.className='pick';
      article.dataset.actionable=String(actionable);
      const status=doc.createElement('span');status.className='pick-status';status.textContent=actionable?'CURRENT':'HISTORICAL / UNVERIFIED';
      const title=doc.createElement('h3');title.textContent=`${safe(item.selection)} ${safe(item.line)}`;
      article.append(status,title);
      appendFacts(article,[
        ['Sport',item.exact_sport],['Market',item.exact_market_family],['Book',item.sportsbook_id],
        ['Odds',item.odds_american],['Observed',formatDate(item.quote_observed_at)],
        ['Analysis',formatDate(item.analysis_generated_at)],['Event starts',formatDate(item.event_start_utc)],
        ['Expires',formatDate(item.expiry_at)],
      ]);
      return article;
    }

    function clearPremium(message,kind='warning'){
      if(state.currentExpiryTimer){timers.clearTimeout(state.currentExpiryTimer);state.currentExpiryTimer=null;}
      const current=element('current-content');
      current.replaceChildren();
      current.textContent='No premium release is present in this page state.';
      current.dataset.state=kind;
      setNotice('current-state',message,kind);
      element('result-filter').disabled=true;
      text('results-content','No entitled result history is present in this page state.');
    }

    function renderSignedOut(message='Sign in to manage an account or view entitled content.'){
      state.authenticated=false;
      state.account=null;
      state.offer=null;
      state.results=[];
      element('sign-in').textContent='Sign in';
      element('sign-in').href='/auth/login?redirect_after='+encodeURIComponent(location.origin+'/');
      element('account-controls').hidden=true;
      text('account-content',message);
      setNotice('account-message','');
      element('refresh-current').hidden=true;
      clearPremium('Authentication is required. No premium content was loaded.','warning');
      text('results-summary','Paper results and actual accepted wagers are reported separately.');
    }

    function describeError(error,context){
      if(error.status===0)return `${context} could not be verified while offline. Previously displayed current picks are no longer actionable.`;
      if(error.status===401)return 'Your session expired. Sign in again; no premium content remains active on this page.';
      if(error.status===403)return `${context} requires an active entitlement for this account.`;
      if(error.status===503){
        const codes=reasonCodes(error.body);
        return `${context} is blocked or temporarily unavailable${codes.length?`: ${codes.join(', ')}`:'.'}`;
      }
      return `${context} is unavailable (HTTP ${error.status}).`;
    }

    async function refreshService(){
      try{
        const status=await request('/status');
        const available=status.service==='available';
        setNotice('service-state',available?'Service available. Checkout remains subject to server launch gates.':'Service is in limited mode. Checkout is disabled, while account and cancellation access remain available.',available?'available':'limited');
      }catch{
        setNotice('service-state','Service status cannot be verified. Checkout should be treated as unavailable.','limited');
      }
    }

    function accountSummary(me){
      const entitlement=me.entitlement;
      const subscription=me.subscription;
      const parts=[];
      parts.push(entitlement?`Active entitlement through ${formatDate(entitlement.expires_at)}.`:'No active entitlement.');
      if(subscription){
        parts.push(`Subscription status: ${safe(subscription.state)}.`);
        parts.push(`Paid through: ${formatDate(subscription.paid_through)}.`);
        if(subscription.cancel_at_period_end)parts.push('Renewal is scheduled to end at the paid-through date.');
      }else parts.push('No billing subscription is on file.');
      return parts.join(' ');
    }

    function renderOffer(offer){
      state.offer=offer;
      const button=element('checkout');
      const product=offer&&offer.product;
      const reasons=offer&&Array.isArray(offer.reason_codes)?offer.reason_codes:[];
      button.disabled=!(offer&&offer.checkout_enabled&&product);
      if(!product){
        setNotice('offer-content','Offer configuration is pending. Checkout is unavailable.','blocked');
        return;
      }
      let price=`${product.amount_minor} minor units ${safe(product.currency)}`;
      try{price=new Intl.NumberFormat(undefined,{style:'currency',currency:product.currency}).format(product.amount_minor/100);}catch{}
      const terms=`Terms ${safe(product.terms_version)}; renewal ${safe(product.renewal_disclosure_version)}; cancellation ${safe(product.cancellation_policy_version)}; refund ${safe(product.refund_policy_version)}.`;
      if(offer.checkout_enabled){
        setNotice('offer-content',`${safe(product.product_code)}: ${price}. ${terms}`,'success');
      }else{
        setNotice('offer-content',`Checkout disabled by server gates${reasons.length?`: ${reasons.join(', ')}`:'.'} ${terms}`,'blocked');
      }
    }

    async function refreshAccount(){
      try{
        const me=await request('/me');
        state.authenticated=true;
        state.account=me;
        element('sign-in').textContent='Account';
        element('sign-in').href='#account';
        element('account-controls').hidden=false;
        element('refresh-current').hidden=false;
        text('account-content',accountSummary(me));
        element('alerts-enabled').checked=Boolean(me.customer&&me.customer.alerts_enabled);
        const hasSubscription=Boolean(me.subscription);
        element('billing-portal').disabled=!hasSubscription;
        element('cancel-subscription').disabled=!hasSubscription||Boolean(me.subscription&&me.subscription.cancel_at_period_end);
        try{renderOffer(await request('/offer'));}catch(error){renderOffer(null);setNotice('offer-content',describeError(error,'Offer'),'blocked');}
        const returned=!state.checkoutReturnHandled&&new URL(location.href).searchParams.has('checkout');
        if(returned){
          state.checkoutReturnHandled=true;
          setNotice('account-message',me.entitlement?'Checkout return verified against an active server entitlement.':'Checkout return received, but the server has not confirmed an entitlement. Access remains pending. ',me.entitlement?'success':'warning');
        }
        return me;
      }catch(error){
        if(error.status===401){renderSignedOut();return null;}
        renderSignedOut(describeError(error,'Account'));
        return null;
      }
    }

    function currentBoundary(payload){
      const values=[payload.expiry_at];
      for(const item of payload.recommendations||[])values.push(item.expiry_at,item.event_start_utc);
      const times=values.map(value=>Date.parse(value)).filter(Number.isFinite);
      return times.length?Math.min(...times):NaN;
    }

    function markCurrentUnverified(message,kind='unverified'){
      setNotice('current-state',message,kind);
      const current=element('current-content');
      current.dataset.state=kind;
      for(const card of current.querySelectorAll('.pick')){
        card.dataset.actionable='false';
        const badge=card.querySelector('.pick-status');
        if(badge)badge.textContent='HISTORICAL / UNVERIFIED';
      }
    }

    function scheduleExpiry(payload){
      if(state.currentExpiryTimer)timers.clearTimeout(state.currentExpiryTimer);
      const boundary=currentBoundary(payload);
      if(!Number.isFinite(boundary)){
        markCurrentUnverified('Release freshness is unknown; the card is not actionable.');
        return;
      }
      const asOf=Date.parse(payload.as_of);
      const serverOffset=Number.isFinite(asOf)?asOf-Date.now():0;
      const remaining=boundary-(Date.now()+serverOffset);
      const expire=()=>{
        markCurrentUnverified('The release reached its expiry or event-start boundary. Revalidating with the server.');
        refreshCurrent('expiry');
      };
      if(remaining<=0){expire();return;}
      state.currentExpiryTimer=timers.setTimeout(expire,Math.min(remaining,2147483647));
    }

    function renderCurrent(payload){
      const current=element('current-content');
      current.replaceChildren();
      if(payload.status!=='CURRENT'){
        const codes=Array.isArray(payload.reason_codes)?payload.reason_codes:[];
        const unavailable=payload.status==='QUALIFIED_FEED_UNAVAILABLE';
        current.textContent=unavailable?'No current verified release. This is a delayed or unavailable feed state, not a normal no-pick day.':'The server reports no current release.';
        current.dataset.state=unavailable?'blocked':'empty';
        setNotice('current-state',codes.length?`Server reason: ${codes.join(', ')}`:'No current release is available.',unavailable?'blocked':'');
        return;
      }
      if(!payload.recommendations||!payload.recommendations.length){
        current.textContent='No qualifying picks in the current verified release.';
        current.dataset.state='empty';
        setNotice('current-state','Verified no-pick release.','success');
        return;
      }
      const grid=doc.createElement('div');grid.className='pick-grid';
      payload.recommendations.forEach(item=>grid.append(renderPick(item,true)));
      current.append(grid);
      current.dataset.state='current';
      setNotice('current-state',`Verified release ${safe(payload.revision_id)}. Published ${formatDate(payload.published_at)}.`,'success');
      scheduleExpiry(payload);
    }

    async function refreshCurrent(trigger='manual'){
      if(!state.authenticated)return;
      const requestNumber=++state.latestCurrentRequest;
      try{
        const payload=await request('/picks/current');
        if(requestNumber!==state.latestCurrentRequest)return;
        const published=Date.parse(payload.published_at);
        if(Number.isFinite(published)&&published<state.acceptedPublishedAt)return;
        if(Number.isFinite(published))state.acceptedPublishedAt=published;
        state.acceptedRevision=payload.revision_id||null;
        renderCurrent(payload);
      }catch(error){
        if(requestNumber!==state.latestCurrentRequest)return;
        if(error.status===401){renderSignedOut('Your session expired. Sign in again.');return;}
        if(error.status===403){
          const current=element('current-content');current.replaceChildren();current.textContent='Signed in, but this account has no active entitlement.';current.dataset.state='not-entitled';
          setNotice('current-state',describeError(error,'Current picks'),'warning');
          return;
        }
        markCurrentUnverified(describeError(error,`Current-pick ${trigger} refresh`));
      }
    }

    function renderResult(item){
      const recommendation=item.recommendation||{};
      const article=doc.createElement('article');article.className='result';article.dataset.status=safe(item.status);
      const status=doc.createElement('span');status.className='pick-status';status.textContent=safe(item.status);
      const title=doc.createElement('h3');title.textContent=`${safe(recommendation.selection)} ${safe(recommendation.line)}`;
      const meta=doc.createElement('p');meta.className='result-meta';meta.textContent=`Recommendation ${safe(item.recommendation_id)} · Release ${safe(item.release_id)} / ${safe(item.revision_id)}`;
      article.append(status,title,meta);
      appendFacts(article,[
        ['Sport',recommendation.exact_sport],['Market',recommendation.exact_market_family],
        ['Original book',recommendation.sportsbook_id],['Original odds',recommendation.odds_american],
        ['Original quote time',formatDate(recommendation.quote_observed_at)],['Original event start',formatDate(recommendation.event_start_utc)],
        ['Paper return',item.paper_return],['Settlement rules',item.settlement_rules_version],
        ['Projection revision',item.projection_revision],['Recorded',formatDate(item.created_at)],
      ]);
      if(item.status==='CORRECTED'||Number(item.projection_revision)>1){
        const correction=doc.createElement('p');correction.className='result-correction';correction.textContent=`Correction/revision ${safe(item.projection_revision)}. Earlier records remain part of the history.`;article.append(correction);
      }
      return article;
    }

    function renderResults(){
      const filter=element('result-filter').value;
      const items=filter==='ALL'?state.results:state.results.filter(item=>item.status===filter);
      const target=element('results-content');target.replaceChildren();
      if(!items.length){target.textContent=state.results.length?'No result records match this filter.':'No result records are available.';return;}
      const list=doc.createElement('div');list.className='results-list';items.forEach(item=>list.append(renderResult(item)));target.append(list);
    }

    async function refreshResults(){
      if(!state.authenticated)return;
      try{
        const payload=await request('/results');
        state.results=Array.isArray(payload.items)?payload.items:[];
        element('result-filter').disabled=false;
        setNotice('results-summary',`${state.results.length} result records. These are paper projections. Actual accepted wagers: ${safe(payload.actual_wagers)}.`,'');
        renderResults();
      }catch(error){
        state.results=[];
        element('result-filter').disabled=true;
        text('results-content',describeError(error,'Results'));
      }
    }

    async function runAction(button,pending,action){
      if(button.disabled)return;
      const original=button.textContent;
      button.disabled=true;button.textContent=pending;
      try{return await action();}
      finally{button.textContent=original;button.disabled=false;}
    }

    async function checkout(){
      const button=element('checkout');
      await runAction(button,'Opening checkout…',async()=>{
        const offer=state.offer;
        if(!offer||!offer.checkout_enabled||!offer.product)throw new ApiError(503,{detail:{reason_codes:['CHECKOUT_NOT_ENABLED']}});
        const product=offer.product;
        try{
          const response=await request('/billing/checkout',{method:'POST',idempotency:true,json:{
            success_url:`${location.origin}/?checkout=return#account`,cancel_url:`${location.origin}/#account`,
            terms_version:product.terms_version,renewal_disclosure_version:product.renewal_disclosure_version,
            cancellation_policy_version:product.cancellation_policy_version,refund_policy_version:product.refund_policy_version,
          }});
          setNotice('account-message','The hosted checkout session was created. Entitlement will still be verified from the server after return.','success');
          location.assign(response.url);
        }catch(error){setNotice('account-message',describeError(error,'Checkout'),'blocked');}
      });
    }

    async function portal(){
      const button=element('billing-portal');
      await runAction(button,'Opening portal…',async()=>{
        try{
          const response=await request('/billing/portal',{method:'POST',idempotency:true,json:{return_url:`${location.origin}/#account`}});
          location.assign(response.url);
        }catch(error){setNotice('account-message',describeError(error,'Billing portal'),'error');}
      });
    }

    async function cancelSubscription(){
      const button=element('cancel-subscription');
      if(!state.cancelArmed){
        state.cancelArmed=true;button.textContent='Confirm cancellation';
        setNotice('account-message','Select Confirm cancellation to request cancellation. Access remains through the confirmed paid-through date.','warning');
        return;
      }
      await runAction(button,'Requesting cancellation…',async()=>{
        try{
          const response=await request('/billing/cancel',{method:'POST',idempotency:true});
          state.cancelArmed=false;
          setNotice('account-message',response.status==='CANCELLATION_PENDING_CONFIRMATION'?'Cancellation requested; provider confirmation is pending.':`Cancellation status: ${safe(response.status)}.`,'success');
          await refreshAccount();
        }catch(error){setNotice('account-message',describeError(error,'Cancellation'),'error');}
      });
      button.textContent='Cancel renewal';
      if(state.account&&state.account.subscription&&state.account.subscription.cancel_at_period_end)button.disabled=true;
    }

    async function logout(){
      const button=element('logout');
      await runAction(button,'Signing out…',async()=>{
        try{await request('/auth/logout',{method:'POST'});}catch(error){setNotice('account-message',describeError(error,'Sign out'),'error');return;}
        renderSignedOut('Signed out.');
      });
    }

    async function saveAlerts(event){
      event.preventDefault();
      const button=element('save-alerts');
      await runAction(button,'Saving…',async()=>{
        const enabled=element('alerts-enabled').checked;
        try{
          const confirmed=await request('/me/alerts',{method:'POST',json:{enabled}});
          element('alerts-enabled').checked=Boolean(confirmed.alerts_enabled);
          if(state.account&&state.account.customer)state.account.customer.alerts_enabled=Boolean(confirmed.alerts_enabled);
          setNotice('account-message',`Alert preference saved: ${confirmed.alerts_enabled?'enabled':'disabled'}.`,'success');
        }catch(error){setNotice('account-message',describeError(error,'Alert preference'),'error');}
      });
    }

    function bind(){
      if(state.bound)return;state.bound=true;
      element('checkout').addEventListener('click',checkout);
      element('billing-portal').addEventListener('click',portal);
      element('cancel-subscription').addEventListener('click',cancelSubscription);
      element('logout').addEventListener('click',logout);
      element('alerts-form').addEventListener('submit',saveAlerts);
      element('refresh-current').addEventListener('click',()=>refreshCurrent('manual'));
      element('result-filter').addEventListener('change',renderResults);
      win.addEventListener('focus',()=>refreshCurrent('focus'));
      win.addEventListener('online',()=>{setNotice('current-state','Connection restored; revalidating current release.','warning');refreshCurrent('reconnect');});
      win.addEventListener('offline',()=>markCurrentUnverified('Offline. Current-release freshness cannot be verified.'));
      doc.addEventListener('visibilitychange',()=>{if(doc.visibilityState==='visible')refreshCurrent('visibility');});
    }

    async function boot(){
      bind();
      renderSignedOut();
      await refreshService();
      const me=await refreshAccount();
      if(me){
        await Promise.all([refreshCurrent('boot'),refreshResults()]);
        state.pollTimer=timers.setInterval(()=>refreshCurrent('poll'),POLL_MS);
      }
      return state;
    }

    return {boot,refreshAccount,refreshCurrent,refreshResults,renderResults,markCurrentUnverified,state,request};
  }

  root.ParlayPickerSubscriber={createSubscriberApp,ApiError,POLL_MS};
  if(root.document){
    root.parlayPickerSubscriberApp=createSubscriberApp();
    root.parlayPickerSubscriberApp.boot();
  }
})(typeof globalThis!=='undefined'?globalThis:this);
