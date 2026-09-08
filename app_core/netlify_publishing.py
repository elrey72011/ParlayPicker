"""Explicit deployment of a sanitized board to one configured Netlify site."""
import io
import re
import zipfile
from urllib.parse import urlsplit
import requests
from scripts.publish_board import render

API = 'https://api.netlify.com/api/v1'


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9-]{8,80}',value):
        raise ValueError('Invalid Netlify identifier')
    return value


def website(value):
    parsed = urlsplit(str(value or ''))
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Netlify did not return a valid HTTPS site URL')
    return str(value)


def archive(package):
    html = render(package)
    output = io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as bundle:
        # Only validated public output is uploaded. Never walk local directories.
        bundle.writestr('index.html',html)
        bundle.writestr('_headers','/*\n  Cache-Control: no-cache\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: no-referrer\n')
    content = output.getvalue()
    if len(content)>10_000_000:
        raise ValueError('Public board exceeds the 10 MB upload limit')
    return content


def api_call(method,path,token,*,data=None):
    if not token:
        raise ValueError('Netlify token is missing')
    headers={'Authorization':'Bearer '+token}
    if data is not None:
        headers['Content-Type']='application/zip'
    try:
        response=requests.request(method,API+path,headers=headers,data=data,timeout=(10,45),allow_redirects=False)
    except requests.RequestException:
        raise RuntimeError('Netlify response unavailable. Check the Netlify dashboard before retrying a deployment.') from None
    if not 200<=response.status_code<300:
        raise RuntimeError(f'Netlify returned HTTP {response.status_code}. Check site access and account limits; no automatic retry was made.')
    try:
        value=response.json()
    except ValueError:
        raise RuntimeError('Netlify returned an unreadable response; check its dashboard before retrying.') from None
    if not isinstance(value,dict):
        raise RuntimeError('Netlify returned an unexpected response')
    return value


def site_info(site_id,token):
    site_id=identifier(site_id)
    value=api_call('GET','/sites/'+site_id,token)
    if value.get('id')!=site_id:
        raise ValueError('Netlify site identity mismatch')
    return {'id':site_id,'url':website(value.get('ssl_url'))}


def deploy(package,site_id,token):
    content=archive(package)
    site_id=identifier(site_id)
    value=api_call('POST','/sites/'+site_id+'/deploys',token,data=content)
    if value.get('site_id')!=site_id:
        raise ValueError('Deployment site mismatch; check the Netlify dashboard')
    return {'id':identifier(value.get('id')),'site_id':site_id,'state':str(value.get('state','processing'))}


def deployment_status(deploy_id,site_id,token):
    value=api_call('GET','/deploys/'+identifier(deploy_id),token)
    if value.get('id')!=deploy_id or value.get('site_id')!=identifier(site_id):
        raise ValueError('Deployment identity mismatch')
    result={'id':deploy_id,'site_id':site_id,'state':str(value.get('state','unknown'))}
    if result['state']=='ready':
        site=api_call('GET','/sites/'+identifier(site_id),token)
        if site.get('id')!=site_id:
            raise ValueError('Netlify site identity mismatch')
        published=site.get('published_deploy') or {}
        if published.get('id')!=deploy_id:
            result['state']='ready_not_published'
        else:
            result['url']=website(site.get('ssl_url'))
    return result
