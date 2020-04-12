from subprocess import Popen, PIPE, STDOUT
import multiprocessing
import os
import signal
import time
import numpy as np

def export_to_file(vs, fname):
    N = vs.shape[0]
    dim = vs.shape[1]
    
    with open(fname, 'w') as fout:
        fout.write('%d\n%d\n'%(dim,N))
        for idx in range(N):
            for d in range(dim):
                fout.write('%f ' % vs[idx,d])
            fout.write('\n')

def export_to_string(vs):
    N = vs.shape[0]
    dim = vs.shape[1]
    
    rets = '%d\n%d\n'%(dim,N)
    for idx in range(N):
        for d in range(dim):
            rets += '%f ' % vs[idx,d]
        rets += '\n'
    return rets
    
def run_qhull_fake(vs,qhull_path, qhull_option):
    timeout = 3
    p = Popen(['/bin/sleep','6000'], stdout=PIPE, stdin=PIPE)
    print ('invoke qhull w #pnts/#dim = ', vs.shape, '  pid =',p.pid, 'timeout=',timeout)
    vstr = export_to_string(vs)
    watchdog = StartWatchdog(timeout, p.pid)

    try:
        stdout_data = p.communicate(input=vstr.encode(), timeout=10)[0]
    except Exception as e:
        p.terminate()
        raise e
    finally:
        StopWatchdog(watchdog)

    output_array = stdout_data.split()
    dim = int(output_array[0])
    N = int(output_array[1])
    assert (dim == vs.shape[1]+1)
    assert (len(output_array) == N*dim + 2)
    return (np.array([float(v) for v in output_array[2:]])).reshape((N,dim))


def SubprocJob(t,pid):
    time.sleep(t)
    print ('Killing', pid,'after timeout =',t)
    os.kill(pid,signal.SIGTERM)
    exit(0)

# parent process - job control
def StartWatchdog(t,pid):
    # sleep time
    # trigger kill
    p = multiprocessing.Process(target=SubprocJob, args = (t,pid))
    p.start()
    return p

def StopWatchdog(watchdog):
    # kill watchdog_pid
    if watchdog.is_alive():
        watchdog.terminate()
    watchdog.join()


def run_qhull(vs, qhull_path, qhull_option, timeout):
    vstr = export_to_string(vs)
    p = Popen([qhull_path] + qhull_option.split() + [ 'n' ], stdout=PIPE, stdin=PIPE)
    print ('invoke qhull w #pnts/#dim = ', vs.shape, '  pid =',p.pid, 'timeout=',timeout)
    
    #watchdog = StartWatchdog(timeout, p.pid)
    try:
        stdout_data = p.communicate(input=vstr.encode(), timeout=timeout)[0]
    except Exception as e:
        p.terminate()
        raise e
    #finally:
    #    StopWatchdog(watchdog)

    output_array = stdout_data.split()
    dim = int(output_array[0])
    N = int(output_array[1])
    assert (dim == vs.shape[1]+1)
    assert (len(output_array) == N*dim + 2)
    return (np.array([float(v) for v in output_array[2:]])).reshape((N,dim))
    
def test():
    v6 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
    H_resp = run_qhull(v6, '/home/hongce/summer19/qhull-2019.1/build/qhull', qhull_option = 'Qt' , timeout = 30)
    print (H_resp)
    #hull = ConvexHull(points=v6)
    #print (hull.equations)

def test2():
    v6 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
    run_qhull_fake(v6,'2','3')

if __name__ == "__main__":
    test2()
    
    

