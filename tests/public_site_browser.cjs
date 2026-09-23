// Run with PLAYWRIGHT_MODULE set to Playwright; BROWSER_CHANNEL defaults to msedge on Windows.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),http=require('node:http'),assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const baseHtml=fs.readFileSync(process.argv[2],'utf8');
const stamp='2026-09-17T16:00:00Z';
const row={id:'game',sport:'MLB',game:'Away at Home',pick:'Home -1.5',market:'spread_home',odds:-110,win_estimate:.6,ev:.1,break_even_probability:.524,status:'APPROVED',quote_source:'Novig',quote_time:stamp,as_of:stamp,start:'2026-09-17T23:00:00Z'};
const initial={schema_version:5,built_at:stamp,stale_after_minutes:30,selection_policy:'qualified-v1',games:{overall:[row],sides:[row],totals:[]},props:[],dfs:[],parlays:[],results:Array.from({length:61},(_,i)=>({id:String(i),date:'2026-09-16',group:'Research',category:'overall',picks:'Saved pick '+i,odds:'-110',outcome:i%2?'WIN':'LOSS',final_score:'4-3'}))};
const screenshotDir=process.env.PUBLIC_SCREENSHOT_DIR;
if(screenshotDir)fs.mkdirSync(screenshotDir,{recursive:true});
async function screenshot(page,name){if(screenshotDir){await page.mouse.move(0,0);if(/sides|totals|research-overall/.test(name)){await page.locator('.pp-view-switcher').evaluate(n=>scrollTo(0,n.getBoundingClientRect().top+scrollY-16));}else{await page.evaluate(()=>scrollTo(0,0));}await page.screenshot({path:screenshotDir+'/'+name+'.png'});}}
function assets(data,publishedAt=stamp){const body=JSON.stringify(data),hash=createHash('sha256').update(body).digest('hex');return {body,version:JSON.stringify({build_id:hash,board_hash:hash,published_at:publishedAt})};}
initial.games.sides=[{...row,game:'Sides matchup',pick:'Home +2.5'}];
initial.games.totals=[{...row,game:'Totals matchup',pick:'Under 8.5',market:'total_under'},{...row,game:'NFL total research',sport:'NFL',status:'PASS',pick:'Over 45.5'}];
initial.research_parlays=[{legs:[{...row,status:'PASS'},{...row,id:'other',game:'Visitors at Hosts',pick:'Under 8.5',status:'PASS'}],status:'RESEARCH',approved_legs:false,win_estimate:.36,decimal_odds_estimate:3.64,ev_estimate:.12}];
const original=assets(initial);let current=original,fail='',boardRequests=0,versionRequests=0;
const html=baseHtml.replace(/(<script id="board-data" type="application\/json">)[\s\S]*?(<\/script>)/,'$1'+JSON.stringify(initial)+'$2').replace(/(<script id="board-version" type="application\/json">)[\s\S]*?(<\/script>)/,'$1'+original.version+'$2');
(async()=>{
 const server=http.createServer((req,res)=>{const path=req.url.split('?')[0];if(path.endsWith('version.json')){versionRequests++;if(fail==='version'){res.writeHead(503);return res.end();}res.setHeader('Content-Type','application/json');return res.end(current.version);}if(path.endsWith('board-data.json')){boardRequests++;if(fail==='board'){res.writeHead(503);return res.end();}return res.end(fail==='hash'?'{}':current.body);}res.setHeader('Content-Type','text/html');res.end(html);});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
 try{
 browser=await chromium.launch({headless:true,...((process.env.BROWSER_CHANNEL||process.platform==='win32')?{channel:process.env.BROWSER_CHANNEL||'msedge'}:{})});
 const page=await browser.newPage({viewport:{width:1280,height:900}});const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));await page.clock.install({time:new Date(stamp)});
 const url='http://127.0.0.1:'+server.address().port+'/saved/';await page.goto(url);await page.waitForFunction(()=>document.getElementById('siteFreshnessText').textContent.includes('Analysis updated'));
 assert.equal(boardRequests,0);assert.ok(versionRequests>=1);
 assert.equal(await page.locator('header').count(),1);assert.equal(await page.locator('[role=tabpanel]').count(),0);
 assert.equal(await page.locator('h1:visible').count(),1);assert.match(await page.locator('#board-title').innerText(),/1 approved play/);
 assert.equal(await page.locator('#approvedPicks .pp-pick-card').count(),1);
 assert.match(await page.locator('#board-title').evaluate(n=>getComputedStyle(n).fontFamily),/ui-serif/);
 assert.match(await page.locator('#approvedPicks .pp-matchup').first().evaluate(n=>getComputedStyle(n).fontFamily),/ui-sans-serif/);
 for(const width of [1280,390]){
  await page.setViewportSize({width,height:900});
  for(const [key,title] of [['games','1 approved play'],['props','Player props'],['parlays','Parlays'],['results','Pick results'],['dfs','DraftKings DFS']]){
   if(width===390&&key==='dfs'){await page.locator('.pp-nav-more summary').click();await page.locator('.pp-nav-more [data-site-tab=dfs]').click();await page.locator('.pp-nav-more summary').click();}else{await page.locator('[data-site-tab="'+key+'"]').first().click();}
   assert.equal(await page.locator('h1:visible').count(),1);assert.equal(await page.locator('h1:visible').innerText(),title);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   if(['props','results','dfs'].includes(key))assert.equal(await page.locator('#'+key+' > h2').filter({hasText:/^(Player Props|Published Pick Results|DraftKings DFS)$/}).count(),0);
   if(key==='parlays')assert.deepEqual(await page.locator('#parlays > h2').allTextContents(),['Parlay products','Legacy qualified combinations','Research Parlays']);
   await screenshot(page,'cleanup-'+key+'-'+width);
  }
  await page.locator('[data-site-tab="games"]').first().click();
  const label=page.locator('.pp-control-label');assert.equal(await label.textContent(),'Research view');assert.equal(await label.isVisible(),true);
  assert.equal(await page.getByRole('group',{name:'Research view'}).count(),1);
  const gap=await page.locator('.pp-research-view-block').evaluate(n=>{const label=n.querySelector('.pp-control-label').getBoundingClientRect(),switcher=n.querySelector('.pp-view-switcher').getBoundingClientRect();return switcher.top-label.bottom;});assert.ok(gap>=0&&gap<=10);
  await screenshot(page,'cleanup-research-overall-'+width);
 }
 await page.setViewportSize({width:1280,height:900});
 assert.equal(await page.locator('#gameBoards h2').innerText(),'Overall Best Picks');
 assert.equal(await page.locator('#gameBoards .pp-card-grid').count(),1);
 assert.equal(await page.locator('#gameBoards .pp-pick-card').count(),1);
 assert.equal(await page.locator('#topPicks').isVisible(),false);
 await page.locator('[data-board-view=sides]').click();assert.equal(await page.locator('[data-board-view=sides]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('[data-board-view=overall]').getAttribute('aria-pressed'),'false');assert.match(await page.locator('#gameBoards').innerText(),/Sides matchup/);
 await screenshot(page,'picks-desktop-sides');
 await page.setViewportSize({width:390,height:844});await screenshot(page,'picks-mobile-sides');assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 await page.setViewportSize({width:1280,height:900});await page.locator('[data-board-view=totals]').click();assert.match(await page.locator('#gameBoards').innerText(),/Totals matchup/);assert.equal(await page.locator('#gameBoards .pp-pick-card').count(),2);await screenshot(page,'picks-desktop-totals');
 await page.selectOption('#pickView','qualified');assert.equal(await page.locator('#gameBoards .pp-pick-card').count(),1);assert.doesNotMatch(await page.locator('#gameBoards').innerText(),/NFL total research/);
 await page.selectOption('#gameLeague','NFL');assert.match(await page.locator('#gameBoards').innerText(),/No qualifying picks in Totals right now/);assert.equal(await page.locator('#approvedPicks .pp-pick-card').count(),1);
 await page.selectOption('#gameLeague','');await page.selectOption('#pickView','research');await page.locator('[data-board-view=overall]').click();
 await page.emulateMedia({colorScheme:'dark'});
 assert.equal(await page.evaluate(()=>getComputedStyle(document.body).backgroundColor),'rgb(244, 239, 231)');
 const active=page.locator('.site-nav a[aria-current=page]').first();
 assert.equal(await active.evaluate(n=>getComputedStyle(n).backgroundColor),'rgba(0, 0, 0, 0)');
 assert.equal(await active.evaluate(n=>getComputedStyle(n).borderBottomColor),'rgb(168, 74, 22)');
 assert.equal(await page.locator('#approvedPicks .pp-odds').first().evaluate(n=>getComputedStyle(n).color),'rgb(28, 36, 48)');
 await page.locator('.site-brand').focus();assert.equal(await page.locator('.site-brand').evaluate(n=>getComputedStyle(n).outlineWidth),'3px');
 await screenshot(page,'picks-desktop-approved');await page.setViewportSize({width:390,height:844});
 await screenshot(page,'picks-mobile-approved');await screenshot(page,'picks-mobile-research-overall');assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 await page.setViewportSize({width:1280,height:900});await page.locator('[data-site-tab="parlays"]').first().click();
 assert.equal(await page.locator('#researchParlayRows .pp-status-research').first().evaluate(n=>getComputedStyle(n).color),'rgb(122, 81, 50)');
 await screenshot(page,'parlays-desktop');

 await page.locator('[data-site-tab="results"]').first().click();await page.selectOption('#resultPeriod','all');await page.selectOption('#resultGroup','Research');assert.equal(await page.locator('#resultDetails table tr').count(),26);await screenshot(page,'results-desktop');await page.getByRole('button',{name:'Load 25 more'}).click();assert.equal(await page.locator('#resultDetails table tr').count(),51);
 await page.locator('#resultSummary details').evaluate(n=>n.open=true);await page.evaluate(()=>scrollTo(0,450));const beforeY=await page.evaluate(()=>scrollY);
 let next=structuredClone(initial);next.results[0].picks='Updated saved pick';current=assets(next);
 await page.clock.fastForward(45000);await page.waitForFunction(()=>window.parlayPickerData.results[0].picks==='Updated saved pick');assert.equal(boardRequests,1);assert.equal(await page.locator('#results').isVisible(),true);assert.equal(await page.locator('#resultPeriod').inputValue(),'all');assert.equal(await page.locator('#resultGroup').inputValue(),'Research');assert.equal(await page.locator('#resultDetails table tr').count(),51);assert.equal(await page.locator('#resultSummary details').evaluate(n=>n.open),true);assert.ok(Math.abs((await page.evaluate(()=>scrollY))-beforeY)<5);
 await page.clock.fastForward(45000);assert.equal(boardRequests,1);
 const saved=await page.evaluate(()=>JSON.stringify(window.parlayPickerData));
 for(const failure of ['version','board','hash']){fail=failure;next=structuredClone(initial);next.results[0].picks='Not accepted '+failure;current=assets(next);await page.clock.fastForward(45000);await page.waitForFunction(()=>document.getElementById('siteFreshnessText').textContent.includes('update check unavailable'));assert.equal(await page.evaluate(()=>JSON.stringify(window.parlayPickerData)),saved);}
 fail='';await page.clock.fastForward(45000);await page.waitForFunction(()=>window.parlayPickerData.results[0].picks==='Not accepted hash');
 await page.selectOption('#resultCategory','sides');assert.match(await page.locator('#resultDetails').innerText(),/No results/);await page.selectOption('#resultCategory','overall');assert.equal(await page.locator('#resultDetails table tr').count(),26);
 for(const width of [390,640,1280]){await page.setViewportSize({width,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);}
 await page.setViewportSize({width:390,height:844});assert.equal(await page.locator('#resultDetails .pp-mobile-results article').count(),25);assert.equal(await page.locator('#resultDetails .pp-desktop-table').isVisible(),false);
 await screenshot(page,'results-mobile');
 await page.locator('[data-site-tab="games"]').first().click();await page.selectOption('#gameLeague','MLB');await page.selectOption('#pickView','qualified');await page.locator('[data-board-view=totals]').click();await page.locator('#gameBoards details').first().evaluate(n=>n.open=true);await page.locator('[data-board-view=totals]').focus();await page.evaluate(()=>scrollTo(0,500));const researchY=await page.evaluate(()=>scrollY);
 next=structuredClone(initial);next.games.overall[0].pick='Updated Home -1.5';current=assets(next);await page.clock.fastForward(45000);await page.waitForFunction(()=>window.parlayPickerData.games.overall[0].pick==='Updated Home -1.5');assert.equal(await page.locator('#gameLeague').inputValue(),'MLB');assert.equal(await page.locator('#pickView').inputValue(),'qualified');assert.equal(await page.locator('[data-board-view=totals]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('#gameBoards h2').innerText(),'Totals');assert.equal(await page.evaluate(()=>document.activeElement.id),'board-view-totals');assert.ok(Math.abs((await page.evaluate(()=>scrollY))-researchY)<5);assert.equal(await page.locator('#gameBoards details').first().evaluate(n=>n.open),true);
 await page.clock.fastForward(31*60000);await page.selectOption('#pickView','research');assert.match(await page.locator('#gameBoards').innerText(),/STALE/);assert.equal(await page.locator('#gameBoards .pp-status-approved').count(),0);
 await page.locator('[data-board-view=overall]').click();await page.setViewportSize({width:1280,height:900});await screenshot(page,'picks-zero-approved');
 next=structuredClone(initial);const publication=await page.evaluate(()=>new Date(Date.now()).toISOString());
 next.built_at=publication;for(const rows of Object.values(next.games))for(const r of rows){r.as_of='2026-09-12T16:00:00Z';r.quote_time=publication;}
 current=assets(next,publication);await page.clock.fastForward(45000);
 await page.waitForFunction(()=>document.getElementById('siteFreshnessText').textContent.includes('Analysis 5d old'));
 assert.match(await page.locator('#siteFreshnessText').innerText(),/Analysis 5d old.*site published just now/);
 assert.equal(await page.locator('#siteFreshness').getAttribute('data-state'),'old');
 assert.match(await page.locator('#board-title').innerText(),/No current wagers right now/);
 await screenshot(page,'picks-old-analysis');
 await page.locator('[data-board-view=sides]').click();await page.reload();assert.equal(await page.locator('#gameBoards h2').innerText(),'Sides');
 const blocked=await browser.newPage();await blocked.addInitScript(()=>{Storage.prototype.getItem=()=>{throw Error('blocked')};Storage.prototype.setItem=()=>{throw Error('blocked')};});await blocked.goto(url);assert.equal(await blocked.locator('#gameBoards h2').innerText(),'Overall Best Picks');await blocked.locator('[data-board-view=totals]').click();assert.equal(await blocked.locator('#gameBoards h2').innerText(),'Totals');await blocked.close();
 assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith(url)));console.log('PASS: live updates, fallback/retry, state, pagination, mobile layouts and stale approval');
 }finally{if(browser)await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
