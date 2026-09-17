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
async function scenario({version=next,board=raw,failVersion=false,failBoard=false,failApply=false,live=true}={}) {
 const calls=[],timers=[],applied=[];
 const elements={'board-version':{textContent:JSON.stringify(old)},siteFreshnessText:{},updateToast:{hidden:true}};
 let shown=payload;const win={parlayPicker:{freshness:version=>analysisFreshness(shown,version),apply(data){if(failApply)throw Error('render failure');applied.push(data);shown=data}}};win.self=win;win.top=win;
 const context={window:win,document:{getElementById:id=>elements[id],querySelector:()=>live},location:{href:'https://example.test/board/index.html',origin:'https://example.test',protocol:'https:'},URL,Date,JSON,Array,Uint8Array,TextEncoder,AbortController,crypto:webcrypto,
 setInterval(fn,ms){timers.push({fn,ms});},setTimeout(){return 1;},clearTimeout(){},
 async fetch(url,options){calls.push({url:String(url),options});if(String(url).includes('version.json')){if(failVersion)throw Error('offline');return {ok:true,text:async()=>JSON.stringify(version)}}if(failBoard)throw Error('offline');return {ok:true,text:async()=>board}}};
 vm.runInNewContext(source,context);
 await new Promise(resolve=>setTimeout(resolve,30));
 return {calls,timers,applied,elements,async poll(){await timers.find(t=>t.ms===45000).fn();}};
}
(async()=>{
 let s=await scenario({version:old});assert.equal(s.calls.length,1);assert.equal(s.applied.length,0);
 s=await scenario();assert.equal(s.applied.length,1);assert.equal(s.calls.length,2);assert.equal(s.elements.updateToast.hidden,false);await s.poll();assert.equal(s.calls.length,3);assert.equal(s.applied.length,1);
 assert.ok(s.calls.every(c=>c.url.startsWith('https://example.test/board/')&&c.options.cache==='no-store'&&c.options.redirect==='error'));
 for(const options of [{failVersion:true},{failBoard:true},{board:'{}'},{board:'invalid JSON'},{failApply:true},{version:{...next,board_hash:'invalid'}},{version:{...next,published_at:'2000-01-01T00:00:00Z'}}]){
  s=await scenario(options);assert.equal(s.applied.length,0);assert.match(s.elements.siteFreshnessText.textContent,/Analysis.*update check unavailable/);const before=s.calls.length;await s.poll();assert.ok(s.calls.length>before);assert.equal(s.elements.updateToast.hidden,true);
 }
 s=await scenario({live:false});assert.equal(s.calls.length,0);assert.match(s.elements.siteFreshnessText.textContent,/Preview/);
 console.log('Polling: same/new builds, failures, hash mismatch, retry, relative URLs and preview isolation passed');
})().catch(e=>{console.error(e);process.exitCode=1});
