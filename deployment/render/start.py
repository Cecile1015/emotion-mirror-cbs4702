"""Run a single bounded CPU service on Render's assigned port."""
import os
import uvicorn
if __name__=='__main__':
    uvicorn.run('api.hosted:create_hosted_app',factory=True,host='0.0.0.0',
                port=int(os.getenv('PORT','10000')),workers=1,
                access_log=False,limit_concurrency=12,timeout_keep_alive=5)
