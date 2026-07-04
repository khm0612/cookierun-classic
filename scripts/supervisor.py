"""Marathon supervisor: keeps ai_farm.py going until TARGET runs complete, surviving the
intermittent native crash (process dies with no traceback, suspected dxcam/DXGI + CUDA
interaction). On a crash it relaunches with the remaining count — ensure_running re-attaches
to any orphaned live run. Stops for: card game (child waits forever by design — supervisor
just keeps waiting too), or two consecutive crashes with zero completed runs (hard fault).
"""
import subprocess, sys, time

from _runtime import ROOT

SCRIPT = ROOT / "scripts" / "ai_farm.py"


def main(target: int = 15) -> int:
    done = 0
    zero_streak = 0
    attempt = 0
    t0 = time.time()
    while done < target:
        attempt += 1
        want = target - done
        print(f"[sup] attempt {attempt}: launching ai_farm for {want} run(s) "
              f"({done}/{target} done, {time.time()-t0:.0f}s elapsed)", flush=True)
        p = subprocess.Popen([sys.executable, "-u", str(SCRIPT), str(want)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1)
        completed = 0
        for line in p.stdout:
            line = line.rstrip()
            print(line, flush=True)
            if line.startswith(">> RESULT:"):
                completed += 1
        rc = p.wait()
        done += completed
        if rc == 0 and completed >= want:
            break
        if completed == 0:
            zero_streak += 1
            if zero_streak >= 2:
                print(f"[sup] two consecutive attempts with ZERO completed runs (rc={rc}) - "
                      "hard fault, stopping. Check the game/emulator state.", flush=True)
                break
        else:
            zero_streak = 0
        print(f"[sup] child exited rc={rc} after {completed} run(s); "
              f"{target-done} remaining - relaunching in 5s", flush=True)
        time.sleep(5)
    print(f"[sup] FINISHED: {done}/{target} runs over {(time.time()-t0)/60:.0f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 15))
