const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const {webcrypto, createHash} = require('node:crypto');
const source = fs.readFileSync('publishing/site.js','utf8');
const template=fs.readFileSync('publishing/board.html','utf8');
const helpers=template.slice(template.indexOf('function analysisTimestamp('),template.indexOf('function probabilityOrder('));
const analysisFreshness=new Function(helpers+';return analysisFreshness;')();
const payload = {schema_version:1,built_at:new Date().toISOString(),stale_after_minutes:30,games:{overall:[],sides:[],totals:[]},props:[],dfs:[]};
const raw = JSON.stringify(payload);
const hash = createHash('sha256').update(raw).digest('hex');
const old = {build_id:'a'.repeat(64),board_hash:'a'.repeat(64),published_at:'2026-01-01T00:00:00Z'};
const next = {build_id:hash,board_hash:hash,published_at:new Date().toISOString()};

let activeScenario;
const cleaned=[];
async function scenario({version=next,board=raw,failVersion=false,failBoard=false,failApply=false,live=true,
 fetchDelayMs=0,hashDelayMs=0,hangFetch=false,hangHash=false,watchdogMs=1000}={}) {
 if(activeScenario)await activeScenario.cleanup();
 const calls=[],timers=[],applied=[],events=[];
 const ownedTimers=new Set(),pending=new Set(),virtualTimers=new Map();
 let nextTimer=1,active=true,renders=0,phase='initialization',resolveInitial;
 const initialDone=new Promise(resolve=>{resolveInitial=resolve});
 const elements={'board-version':{textContent:JSON.stringify(old)},siteFreshnessText:{},updateToast:{hidden:true}};
 // This is the LAST presentation assignment in production renderFreshness.
 // Promise continuations run after the whole synchronous final render returns.
 elements.boardAnalysisAge={set textContent(value){if(active){events.push('final-render');if(++renders===2)resolveInitial();}}};
 function delay(ms,forever=false){
  if(!ms&&!forever)return Promise.resolve();
  return new Promise((resolve,reject)=>{
   let timer;
   const cancel=()=>{if(timer){clearTimeout(timer);ownedTimers.delete(timer);}pending.delete(cancel);reject(Error('Synthetic scenario disposed'));};
   pending.add(cancel);
   if(!forever){timer=setTimeout(()=>{ownedTimers.delete(timer);pending.delete(cancel);resolve();},ms);ownedTimers.add(timer);}
  });
 }
 let shown=payload;
 const win={parlayPicker:{freshness:version=>analysisFreshness(shown,version),apply(data){phase='apply';events.push('apply');if(failApply)throw Error('render failure');applied.push(data);shown=data}}};win.self=win;win.top=win;
 const state={calls,timers,applied,elements,events,
  async poll(){assert.ok(active,'polling a disposed scenario');await timers.find(t=>t.ms===45000).fn();},
  async cleanup(){
   if(!active)return;
   active=false;for(const cancel of [...pending])cancel();
   for(const timer of ownedTimers)clearTimeout(timer);ownedTimers.clear();virtualTimers.clear();timers.length=0;
   // Drain canceled work's catch/finally render, never retry an update.
   await Promise.resolve();await Promise.resolve();await new Promise(resolve=>setImmediate(resolve));
   assert.equal(pending.size,0);assert.equal(ownedTimers.size,0);assert.equal(virtualTimers.size,0);
   cleaned.push({pending:pending.size,timers:ownedTimers.size,virtual:virtualTimers.size});
   if(activeScenario===state)activeScenario=undefined;
  }};
 activeScenario=state;
 const context={window:win,document:{getElementById:id=>elements[id],querySelector:()=>live},location:{href:'https://example.test/board/index.html',origin:'https://example.test',protocol:'https:'},URL,Date,JSON,Array,Uint8Array,TextEncoder,AbortController,
  crypto:{subtle:{async digest(...args){phase='hash';await delay(hashDelayMs,hangHash);const value=await webcrypto.subtle.digest(...args);if(!active)throw Error('Synthetic scenario disposed');events.push('hash-complete');return value;}}},
  setInterval(fn,ms){const id=nextTimer++;timers.push({fn,ms});virtualTimers.set(id,{fn,ms,kind:'interval'});return id;},
  setTimeout(fn,ms){const id=nextTimer++;virtualTimers.set(id,{fn,ms,kind:'timeout'});return id;},
  clearTimeout(id){virtualTimers.delete(id);},
  async fetch(url,options){phase=String(url).includes('version.json')?'version-fetch':'board-fetch';calls.push({url:String(url),options});await delay(fetchDelayMs,hangFetch);
   if(String(url).includes('version.json')){if(failVersion)throw Error('offline');return {ok:true,text:async()=>JSON.stringify(version)}}
   if(failBoard)throw Error('offline');return {ok:true,text:async()=>board};}};
 let watchdog;
 try{
  vm.runInNewContext(source,context);
  if(live)await Promise.race([initialDone,new Promise((_,reject)=>{watchdog=setTimeout(()=>reject(Error(`Initial public-refresh update unfinished: phase=${phase}; calls=${calls.length}; finalRenders=${renders}; watchdogMs=${watchdogMs}`)),watchdogMs);ownedTimers.add(watchdog);})]);
  return state;
 }catch(error){await state.cleanup();error.message+='; cleanup=complete';throw error;}
 finally{if(watchdog){clearTimeout(watchdog);ownedTimers.delete(watchdog);}}
}
async function originalSuite(){

 let s=await scenario({version:old});assert.equal(s.calls.length,1);assert.equal(s.applied.length,0);
 s=await scenario();assert.equal(s.applied.length,1);assert.equal(s.calls.length,2);assert.equal(s.elements.updateToast.hidden,false);await s.poll();assert.equal(s.calls.length,3);assert.equal(s.applied.length,1);
 assert.ok(s.calls.every(c=>c.url.startsWith('https://example.test/board/')&&c.options.cache==='no-store'&&c.options.redirect==='error'));
 for(const options of [{failVersion:true},{failBoard:true},{board:'{}'},{board:'invalid JSON'},{failApply:true},{version:{...next,board_hash:'invalid'}},{version:{...next,published_at:'2000-01-01T00:00:00Z'}}]){
  s=await scenario(options);assert.equal(s.applied.length,0);assert.match(s.elements.siteFreshnessText.textContent,/Analysis.*update check unavailable/);const before=s.calls.length;await s.poll();assert.ok(s.calls.length>before);assert.equal(s.elements.updateToast.hidden,true);
 }
 s=await scenario({live:false});assert.equal(s.calls.length,0);assert.match(s.elements.siteFreshnessText.textContent,/Preview/);
 console.log('Polling: same/new builds, failures, hash mismatch, retry, relative URLs and preview isolation passed');

}
function versionFor(body){const h=createHash('sha256').update(body).digest('hex');return {...next,build_id:h,board_hash:h};}
async function controls(name){
 try{
  if(name==='delayed-fetch-hash'){
   const s=await scenario({fetchDelayMs:70,hashDelayMs:80});
   assert.equal(s.applied.length,1);assert.ok(s.events.indexOf('hash-complete')<s.events.indexOf('apply'));
   assert.equal(s.events.at(-1),'final-render');assert.equal(s.elements.updateToast.hidden,false);
   await s.poll();assert.equal(s.applied.length,1);
  }else if(name==='rejections'){
   for(const body of ['invalid JSON',JSON.stringify({...payload,games:null})]){
    const s=await scenario({board:body,version:versionFor(body),fetchDelayMs:40,hashDelayMs:60});
    assert.equal(s.applied.length,0);assert.match(s.elements.siteFreshnessText.textContent,/update check unavailable/);assert.equal(s.events.at(-1),'final-render');
   }
   const s=await scenario({failApply:true,hashDelayMs:80});assert.equal(s.applied.length,0);assert.match(s.elements.siteFreshnessText.textContent,/update check unavailable/);
  }else if(name==='never-fetch'||name==='never-hash'){
   await scenario({hangFetch:name==='never-fetch',hangHash:name==='never-hash',watchdogMs:100});
   assert.fail('Unfinished work must not succeed');
  }else if(name==='cleanup'){
   const first=await scenario({hashDelayMs:60});await first.cleanup();
   await assert.rejects(first.poll(),/disposed/);
   await assert.rejects(scenario({hangHash:true,watchdogMs:60}),/unfinished.*cleanup=complete/);
   const next=await scenario({failBoard:true});assert.equal(next.applied.length,0);await next.cleanup();
   assert.equal(activeScenario,undefined);assert.ok(cleaned.every(x=>x.pending===0&&x.timers===0&&x.virtual===0));
  }else if(name==='assertion'){
   assert.equal('synthetic actual','synthetic expected','deliberate synthetic JavaScript assertion');
  }else throw Error('Unknown synthetic control');
  console.log('Synthetic control passed: '+name);
 }finally{if(activeScenario)await activeScenario.cleanup();}
}
if(require.main===module){
 (process.argv[2]==='--control'?controls(process.argv[3]):originalSuite())
  .catch(error=>{console.error(error);process.exitCode=1;})
  .finally(async()=>{if(activeScenario)await activeScenario.cleanup();});
}
module.exports={scenario,originalSuite,controls};
