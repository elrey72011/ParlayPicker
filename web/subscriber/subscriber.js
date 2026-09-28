const api='/api/v1';
const text=(id,value)=>{document.getElementById(id).textContent=value};
const safe=(value)=>String(value??'Unavailable');
async function get(path){const r=await fetch(api+path,{credentials:'include',headers:{Accept:'application/json'}});if(!r.ok)throw new Error(String(r.status));return r.json()}
function renderPick(item){const article=document.createElement('article');article.className='pick';const title=document.createElement('h3');title.textContent=`${safe(item.selection)} ${safe(item.line)}`;article.append(title);const dl=document.createElement('dl');dl.className='facts';for(const [label,key] of [['Sport','exact_sport'],['Market','exact_market_family'],['Book','sportsbook_id'],['Odds','odds_american'],['Observed','quote_observed_at'],['Expires','expiry_at']]){const dt=document.createElement('dt');dt.textContent=label;const dd=document.createElement('dd');dd.textContent=safe(item[key]);dl.append(dt,dd)}article.append(dl);return article}
async function boot(){
  document.getElementById('sign-in').href='/auth/login?redirect_after='+encodeURIComponent(location.origin+'/');
  try{const status=await get('/status');text('service-state',status.service==='available'?'Service available. Checkout remains subject to launch gates.':'Service is in limited mode; paid checkout is disabled.')}catch{text('service-state','Service status is temporarily unavailable.')}
  try{
    const me=await get('/me');text('account-content',me.entitlement?`Entitled through ${safe(me.entitlement.expires_at)}. Subscription: ${safe(me.subscription?.state)}.`:'No active entitlement. Account and cancellation support remain available.');document.getElementById('sign-in').textContent='Account';document.getElementById('sign-in').href='#account';
    try{const current=await get('/picks/current');const target=document.getElementById('current-content');target.replaceChildren();if(!current.recommendations?.length){target.textContent=current.status==='QUALIFIED_FEED_UNAVAILABLE'?'No current verified release. This is not presented as a normal no-pick day.':'No qualifying picks in the current verified release.'}else{const grid=document.createElement('div');grid.className='pick-grid';current.recommendations.forEach(item=>grid.append(renderPick(item)));target.append(grid)}}catch{text('current-content','Current picks are unavailable for this account.')}
    try{const results=await get('/results');text('results-content',`${results.items.length} result records. Actual accepted wagers: ${results.actual_wagers}.`)}catch{text('results-content','Results are unavailable for this account.')}
  }catch{/* logged out: public shell intentionally contains no premium payload */}
}
boot();
