const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs'),http=require('node:http'),assert=require('node:assert/strict');

const [htmlPath,jsPath,cssPath]=process.argv.slice(2);
const html=fs.readFileSync(htmlPath,'utf8'),js=fs.readFileSync(jsPath,'utf8'),css=fs.readFileSync(cssPath,'utf8');
const now='2026-09-28T16:00:00.000Z';
const recommendation={schema_version:2,recommendation_id:'rec-1',exact_sport:'NFL',exact_market_family:'SPREAD',canonical_event_id:'event-1',selection:'Example +3.5',line:3.5,sportsbook_id:'book-test',odds_american:-110,odds_decimal:1.91,quote_observed_at:'2026-09-28T15:55:00Z',analysis_generated_at:'2026-09-28T15:56:00Z',event_start_utc:'2026-09-28T16:30:00Z',expiry_at:'2026-09-28T16:02:00Z',probability_semantics:'win_unconditional_with_push',p_win:.55,p_push:0,p_loss:.45,mean_ev_per_unit:.0505,p_win_conservative:.53,conservative_ev_per_unit:.0123,uncertainty_method:'fixed_push_lower_win_bound',minimum_acceptable_decimal_odds:1.9,minimum_acceptable_line:3.5,disclosure_version:'disclosure-v1'};
const releases={
  old:{schema_version:2,status:'CURRENT',release_id:'release-old',revision_id:'revision-old',published_at:'2026-09-28T15:58:00Z',expiry_at:'2026-09-28T16:25:00Z',as_of:now,content_hash:'a'.repeat(64),recommendations:[{...recommendation,selection:'Old response +3.5',expiry_at:'2026-09-28T16:25:00Z'}]},
  current:{schema_version:2,status:'CURRENT',release_id:'release-1',revision_id:'revision-1',published_at:'2026-09-28T15:59:00Z',expiry_at:recommendation.expiry_at,as_of:now,content_hash:'b'.repeat(64),recommendations:[recommendation]},
  newer:{schema_version:2,status:'CURRENT',release_id:'release-2',revision_id:'revision-2',published_at:'2026-09-28T16:01:00Z',expiry_at:'2026-09-28T16:25:00Z',as_of:'2026-09-28T16:01:00Z',content_hash:'c'.repeat(64),recommendations:[{...recommendation,selection:'New response +4.0',line:4,expiry_at:'2026-09-28T16:25:00Z'}]},
  unavailable:{schema_version:1,status:'QUALIFIED_FEED_UNAVAILABLE',as_of:now,release_id:null,recommendations:[],reason_codes:['NO_CURRENT_VERIFIED_RELEASE'],retryable:false},
};
const resultStatuses=['WIN','LOSS','PUSH','VOID','PENDING','NEEDS_REVIEW','CORRECTED'];
const results=resultStatuses.map((status,index)=>({recommendation_id:`rec-${index+1}`,status,paper_return:status==='WIN'?0.91:status==='LOSS'?-1:status==='PUSH'||status==='VOID'?0:null,settlement_rules_version:'settle-v1',projection_revision:status==='CORRECTED'?2:1,created_at:`2026-09-2${index+1}T12:00:00Z`,release_id:'release-history',revision_id:'revision-history',release_status:'ACTIVE_REVISION_PROMOTED',promoted_at:'2026-09-20T12:00:00Z',expires_at:'2026-09-20T15:00:00Z',recommendation:{...recommendation,recommendation_id:`rec-${index+1}`,selection:`${status} selection`,quote_observed_at:'2026-09-20T11:00:00Z',event_start_utc:'2026-09-20T15:00:00Z',expiry_at:'2026-09-20T14:30:00Z'}}));

let validSession=true,entitled=false,subscription=false,salesEnabled=true,currentMode='current',raceCount=0;
let checkoutPosts=0,portalPosts=0,cancelPosts=0,logoutPosts=0,alertPosts=0,currentGets=0,resultGets=0;
const observedPaths=[],mutationHeaders=[];
function json(res,status,body,headers={}){res.writeHead(status,{'Content-Type':'application/json',...headers});res.end(JSON.stringify(body));}
function hasSession(req){return validSession&&/pp_session=session-token/.test(req.headers.cookie||'');}
function mutationAllowed(req){mutationHeaders.push({'csrf':req.headers['x-csrf-token'],'idempotency':req.headers['idempotency-key']||null,path:req.url});return hasSession(req)&&req.headers['x-csrf-token']==='csrf-token';}
async function body(req){let raw='';for await(const chunk of req)raw+=chunk;return raw?JSON.parse(raw):null;}

(async()=>{
  const server=http.createServer(async(req,res)=>{
    const path=new URL(req.url,'http://test').pathname;observedPaths.push(path);
    if(path==='/subscriber.js'){res.writeHead(200,{'Content-Type':'text/javascript'});return res.end(js);}
    if(path==='/subscriber.css'){res.writeHead(200,{'Content-Type':'text/css'});return res.end(css);}
    if(path==='/auth/login')return void json(res,303,{}, {'Location':'/','Set-Cookie':['pp_session=session-token; Path=/; HttpOnly; SameSite=Lax','pp_csrf=csrf-token; Path=/; SameSite=Strict']});
    if(path==='/auth/logout'&&req.method==='POST'){
      logoutPosts++;if(!mutationAllowed(req))return void json(res,403,{detail:'CSRF validation failed'});
      validSession=false;return void json(res,204,null,{'Set-Cookie':['pp_session=; Max-Age=0; Path=/','pp_csrf=; Max-Age=0; Path=/']});
    }
    if(path==='/api/v1/status')return void json(res,200,{schema_version:1,service:'available',sales:salesEnabled?'enabled':'disabled',source_revision:'browser-fixture',as_of:now});
    if(path.startsWith('/api/v1/')&&!hasSession(req))return void json(res,401,{detail:'authentication required'});
    if(path==='/api/v1/me')return void json(res,200,{customer:{id:'customer-1',contact_email:'customer@example.test',email_verified:true,status:'ACTIVE',role:'CUSTOMER',alerts_enabled:true,created_at:now},entitlement:entitled?{product_code:'monthly-qualified-straights',expires_at:'2026-10-28T16:00:00Z'}:null,subscription:subscription?{state:'ACTIVE',paid_through:'2026-10-28T16:00:00Z',cancel_at_period_end:cancelPosts>0,updated_at:now}:null});
    if(path==='/api/v1/offer')return void json(res,200,{schema_version:1,checkout_enabled:salesEnabled,reason_codes:salesEnabled?[]:['SALES_NOT_OWNER_ENABLED'],evaluated_at:now,product:{product_code:'monthly-qualified-straights',version:1,currency:'USD',amount_minor:1000,selling_entity:'Test Entity',offered_markets:['NFL:SPREAD'],eligible_jurisdictions:['TEST'],terms_version:'terms-v1',renewal_disclosure_version:'renewal-v1',cancellation_policy_version:'cancel-v1',refund_policy_version:'refund-v1'}});
    if(path==='/api/v1/picks/current'){
      currentGets++;if(!entitled)return void json(res,403,{detail:'active entitlement required'});
      if(currentMode==='fail')return void json(res,503,{detail:{status:'RELEASE_BLOCKED',reason_codes:['UPSTREAM_UNAVAILABLE']}});
      if(currentMode==='race'){
        raceCount++;if(raceCount===1)return void setTimeout(()=>json(res,200,releases.old),250);
        return void json(res,200,releases.newer);
      }
      return void json(res,200,releases[currentMode]);
    }
    if(path==='/api/v1/results'){
      resultGets++;if(!entitled)return void json(res,403,{detail:'active entitlement required'});
      return void json(res,200,{schema_version:1,paper_results:true,actual_wagers:'UNAVAILABLE',items:results});
    }
    if(path==='/api/v1/me/alerts'&&req.method==='POST'){
      alertPosts++;if(!mutationAllowed(req))return void json(res,403,{detail:'CSRF validation failed'});
      const supplied=await body(req);return void json(res,200,{alerts_enabled:Boolean(supplied.enabled)});
    }
    if(path==='/api/v1/billing/checkout'&&req.method==='POST'){
      checkoutPosts++;if(!mutationAllowed(req))return void json(res,403,{detail:'CSRF validation failed'});
      const supplied=await body(req);assert.equal(supplied.terms_version,'terms-v1');assert.equal(supplied.refund_policy_version,'refund-v1');
      subscription=true;return void setTimeout(()=>json(res,200,{checkout_session_id:'cs_test',url:`http://127.0.0.1:${server.address().port}/?checkout=return#account`}),100);
    }
    if(path==='/api/v1/billing/cancel'&&req.method==='POST'){
      cancelPosts++;if(!mutationAllowed(req))return void json(res,403,{detail:'CSRF validation failed'});
      return void setTimeout(()=>json(res,202,{status:'CANCELLATION_PENDING_CONFIRMATION'}),100);
    }
    if(path==='/api/v1/billing/portal'&&req.method==='POST'){
      portalPosts++;if(!mutationAllowed(req))return void json(res,403,{detail:'CSRF validation failed'});
      return void json(res,200,{url:`http://127.0.0.1:${server.address().port}/portal-destination`});
    }
    if(path==='/portal-destination'){res.writeHead(200,{'Content-Type':'text/html'});return res.end('<h1>Hosted billing portal fixture</h1>');}
    res.writeHead(200,{'Content-Type':'text/html'});res.end(html);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`;let browser;
  try{
    browser=await chromium.launch({headless:true,...(process.platform==='win32'?{channel:process.env.BROWSER_CHANNEL||'msedge'}:{})});
    const page=await browser.newPage({viewport:{width:1280,height:900}});const pageErrors=[];page.on('pageerror',error=>pageErrors.push(error.message));
    await page.clock.install({time:new Date(now)});

    await page.goto(base+'/');
    await page.waitForFunction(()=>document.querySelector('#service-state').textContent.includes('Service available'));
    assert.match(await page.locator('#account-content').innerText(),/Sign in/);
    assert.match(await page.locator('#current-content').innerText(),/No premium release/);
    assert.equal(currentGets,0);assert.equal(resultGets,0);
    const direct=await page.request.get(base+'/api/v1/picks/current');assert.equal(direct.status(),401);

    await page.locator('#sign-in').click();await page.waitForURL(base+'/');await page.waitForFunction(()=>document.querySelector('#account-content').textContent.includes('No active entitlement'));
    assert.equal(await page.locator('#checkout').isEnabled(),true);
    await page.locator('#checkout').dblclick();await page.waitForURL(/checkout=return/);
    await page.waitForFunction(()=>document.querySelector('#account-message').textContent.includes('has not confirmed an entitlement'));
    assert.equal(checkoutPosts,1);assert.match(await page.locator('#current-content').innerText(),/no active entitlement/i);

    entitled=true;await page.reload();await page.waitForFunction(()=>document.querySelector('#current-content').textContent.includes('Example +3.5'));
    assert.match(await page.locator('#account-content').innerText(),/Active entitlement/);
    assert.equal(await page.locator('#results-content .result').count(),7);
    for(const status of resultStatuses)assert.equal(await page.locator(`#results-content .result[data-status="${status}"]`).count(),1);
    assert.match(await page.locator('#results-summary').innerText(),/paper projections.*UNAVAILABLE/i);
    assert.match(await page.locator('#results-content').innerText(),/Original quote time/);
    assert.match(await page.locator('#results-content').innerText(),/Correction\/revision 2/);
    await page.selectOption('#result-filter','LOSS');assert.equal(await page.locator('#results-content .result').count(),1);
    await page.selectOption('#result-filter','ALL');

    await page.uncheck('#alerts-enabled');await page.locator('#save-alerts').click();
    await page.waitForFunction(()=>document.querySelector('#account-message').textContent.includes('disabled'));
    assert.equal(alertPosts,1);assert.equal(mutationHeaders.at(-1).csrf,'csrf-token');

    currentMode='unavailable';await page.clock.fastForward(121000);
    await page.waitForFunction(()=>document.querySelector('#current-content').textContent.includes('No current verified release'));
    assert.match(await page.locator('#current-state').innerText(),/NO_CURRENT_VERIFIED_RELEASE/);

    currentMode='current';await page.evaluate(()=>window.parlayPickerSubscriberApp.refreshCurrent('restore'));
    await page.waitForFunction(()=>document.querySelector('#current-content').textContent.includes('Example +3.5'));
    await page.evaluate(()=>dispatchEvent(new Event('offline')));
    assert.equal(await page.locator('#current-content .pick').getAttribute('data-actionable'),'false');
    assert.match(await page.locator('#current-state').innerText(),/Offline/);

    currentMode='race';raceCount=0;
    await page.evaluate(()=>{window.parlayPickerSubscriberApp.refreshCurrent('late-old');setTimeout(()=>window.parlayPickerSubscriberApp.refreshCurrent('newer'),10);});
    await page.waitForFunction(()=>document.querySelector('#current-content').textContent.includes('New response +4.0'));
    await page.waitForTimeout(350);assert.doesNotMatch(await page.locator('#current-content').innerText(),/Old response/);

    currentMode='unavailable';await page.evaluate(()=>dispatchEvent(new Event('focus')));
    await page.waitForFunction(()=>document.querySelector('#current-content').textContent.includes('No current verified release'));

    salesEnabled=false;await page.reload();await page.waitForFunction(()=>document.querySelector('#offer-content').textContent.includes('SALES_NOT_OWNER_ENABLED'));
    assert.equal(await page.locator('#billing-portal').isEnabled(),true);assert.equal(await page.locator('#cancel-subscription').isEnabled(),true);
    await page.locator('#cancel-subscription').click();assert.equal(cancelPosts,0);assert.match(await page.locator('#account-message').innerText(),/Confirm cancellation/);
    await page.locator('#cancel-subscription').click();await page.waitForFunction(()=>document.querySelector('#account-message').textContent.includes('pending'));
    assert.equal(cancelPosts,1);
    await page.locator('#billing-portal').click();await page.waitForURL(/portal-destination/);assert.equal(portalPosts,1);await page.goBack();
    await page.waitForFunction(()=>document.querySelector('#logout')&&!document.querySelector('#account-controls').hidden);

    await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.locator('#logout').focus();await page.keyboard.press('Enter');await page.waitForFunction(()=>document.querySelector('#account-content').textContent.includes('Signed out'));
    assert.equal(logoutPosts,1);assert.match(await page.locator('#current-content').innerText(),/No premium release/);
    validSession=false;await page.goto(base+'/?checkout=success');await page.waitForFunction(()=>document.querySelector('#account-content').textContent.includes('Sign in'));
    assert.match(await page.locator('#current-content').innerText(),/No premium release/);

    assert.ok(mutationHeaders.every(entry=>entry.csrf==='csrf-token'));
    assert.ok(mutationHeaders.filter(entry=>entry.path.includes('/billing/')).every(entry=>entry.idempotency));
    assert.deepEqual(pageErrors,[]);
    assert.ok(observedPaths.every(path=>path.startsWith('/')&&!/drive|sportsbook|analysis-job/.test(path)));
    console.log(JSON.stringify({status:'PASS',context:'LOCAL_INTEGRATION',checkoutPosts,portalPosts,cancelPosts,logoutPosts,alertPosts,currentGets,resultGets,resultStatuses,lateResponseSuppressed:true,expiryCleared:true,mobileKeyboard:true,premiumLeakage:false}));
  }finally{if(browser)await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
