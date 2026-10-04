import os,sys,json
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,"/home/claude/s3/code")
import run_s3
from multiprocessing import Pool
OUT="/home/claude/s3linux/results_conf_linux.jsonl"
jobs=[j for j in run_s3.design() if j["stage"]=="main" and j["regime"]=="noniid" and j["config"] in ("FG","B4","B5")]
done=set()
if os.path.exists(OUT):
    done={run_s3.key(json.loads(l)) for l in open(OUT)}
todo=[j for j in jobs if run_s3.key(j) not in done]
print(len(jobs),"jobs",len(todo),"todo",flush=True)
if __name__=="__main__":
    with Pool(int(sys.argv[1]) if len(sys.argv)>1 else 2) as p:
        for r in p.imap_unordered(run_s3.work,todo):
            r.pop("loss_curve",None)
            open(OUT,"a").write(json.dumps(r)+"\n")
