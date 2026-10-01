import os,sys,json
os.environ['DATABASE_URL']='sqlite:///./smoke_r.db'
os.environ['DEMO_MODE']='true'
os.environ['INLINE_WORKER']='false'
sys.path.insert(0,'.')
from fastapi.testclient import TestClient
from app.main import app
from app.database import Base,engine,SessionLocal
from app.seed import seed_demo
Base.metadata.drop_all(engine); Base.metadata.create_all(engine)
with SessionLocal() as db: seed_demo(db)
c=TestClient(app)
p=json.dumps({'lines_added':1.0,'lines_deleted':1.0,'files_changed':1.0,'code_churn':1.0,'commit_entropy':1.0,'developer_experience':1.0,'file_age_days':1.0,'commit_frequency':1.0,'contributor_count':1.0,'complexity':1.0,'previous_defects':1.0})
for method,path,body in [('GET','/health',None),('GET','/auth/status',None),('GET','/auth/me',None),('GET','/repositories',None),('GET','/analyses',None),('GET','/models/current/metrics',None),('GET','/webhooks/health',None),('POST','/predict',p),('POST','/auth/logout',None)]:
    r=c.get(path) if method=='GET' else (c.post(path,content=body,headers={'Content-Type':'application/json'}) if body else c.post(path))
    ok=r.status_code<400
    print(('PASS' if ok else 'FAIL'),' ',method,path,'HTTP',r.status_code,'' if ok else r.text[:150])
