// Verify an actual generated bundle without replacing its analysis with fixtures.
// Usage: node tests/public_bundle_browser.cjs path/to/index.html
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const assert=require('node:assert/strict'),{createHash}=require('node:crypto');
const root=path.dirname(path.resolve(process.argv[2]));
const board=fs.readFileSync(path.join(root,'board-data.json'));
const expected=JSON.parse(board),version=JSON.parse(fs.readFileSync(path.join(root,'version.json')));
assert.equal(createHash('sha256').update(board).digest('hex'),version.board_hash);
assert.equal(version.build_id,version.board_hash);
(async()=>{
 const server=http.createServer((req,res)=>{
  const name=req.url.split('?')[0].split('/').pop()||'index.html';
  if(!['index.html','board-data.json','version.json','site.css','site.js'].includes(name)){res.writeHead(404);return res.end();}
  res.setHeader('Content-Type',name.endsWith('.json')?'application/json':name.endsWith('.html')?'text/html':'text/plain');
  res.end(fs.readFileSync(path.join(root,name)));
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
 try{
  browser=await chromium.launch({headless:true,...((process.env.BROWSER_CHANNEL||process.platform==='win32')?{channel:process.env.BROWSER_CHANNEL||'msedge'}:{})});
  const page=await browser.newPage({viewport:{width:1280,height:900},colorScheme:'dark'});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('http://127.0.0.1:'+server.address().port+'/');
  await page.waitForFunction(()=>window.parlayPickerData);
  assert.deepEqual(await page.evaluate(()=>window.parlayPickerData),expected);
  assert.deepEqual(await page.locator('#board-version').evaluate(n=>JSON.parse(n.textContent)),version);
  assert.equal(await page.evaluate(()=>getComputedStyle(document.body).backgroundColor),'rgb(244, 239, 231)');
  assert.equal(await page.locator('h1:visible').count(),1);
  assert.equal(await page.locator('.pp-control-label').textContent(),'Research view');
  for(const width of [1280,390]){
   await page.setViewportSize({width,height:900});
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   await page.screenshot({path:path.join(root,'actual-public-'+width+'.png')});
   for(const view of ['overall','sides','totals']){
    await page.locator('[data-board-view='+view+']').click();
    assert.equal(await page.locator('[data-board-view='+view+']').getAttribute('aria-pressed'),'true');
    assert.equal(await page.locator('#gameBoards h2').count(),1);
   }
   await page.locator('[data-board-view=overall]').click();
   await page.evaluate(()=>scrollTo(0,0));
  }
  assert.deepEqual(errors,[]);
  console.log('PASS: actual package '+expected.built_at+', matching embedded data/manifest, light theme, research views, desktop/mobile');
 }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
