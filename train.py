import os
import subprocess
import argparse
from datetime import datetime
import time
import re
import sys

def get_simple_status(log_file):
    """로그 파일에서 간단한 상태 정보만 추출"""
    status = {
        'current_iter': 0,
        'best_dice': 0.0
    }

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 마지막 100줄만 검사 (성능 최적화)
        for line in lines[-100:]:
            # iteration 추출
            iter_match = re.search(r'iteration[:\s]+(\d+)', line, re.IGNORECASE)
            if iter_match:
                status['current_iter'] = int(iter_match.group(1))

            # Best dice 추출
            best_match = re.search(r'Best-[ST]:\s*([\d.]+)', line, re.IGNORECASE)
            if best_match:
                status['best_dice'] = float(best_match.group(1))
    except:
        pass

    return status

def create_experiment_configs(use_poly=False):
    """
    실험 설정 정의
    """
    exp_name = "A1TCP"
    experiments = [
        {
            "name": f"{exp_name}_44_v2",
            "gpu_id": 0,
            "script": "A1TCP.py",
            "labeled_num": 44,
            "max_iterations": 30000,
            "aug_patience": 15,
            "aug_min_delta": 0.01,
            "copy_paste_prob": 1.0,
            "cutmix_prob": 1.0,
            "num_classes": 2,    # [명시적 지정]
            "poly": use_poly,    # [인자로 받은 값 사용]
        }
    ]
    return experiments

def build_command(exp_config, base_path, data_path, python_exe):
    """
    실험 설정으로부터 명령어 생성
    """
    script_path = os.path.join(base_path, "code", exp_config["script"])
    cfg_path = os.path.join(base_path, "cfgs", "config_2d_kvasir_5percent.yml")
    
    cmd = [
        python_exe,
        script_path,
        f"--gpu_id={exp_config['gpu_id']}",
        "--cfg", cfg_path,
        "--root_path", data_path,
        f"--labeled_num={exp_config['labeled_num']}",
        f"--exp=Kvasir/{exp_config['name']}",
        "--model=unet",
        f"--max_iterations={exp_config['max_iterations']}",
        
        # [핵심 수정] 여기서 2를 강제 주입하여 Config 파일의 설정을 무시하게 함
        f"--num_classes={exp_config.get('num_classes', 2)}" 
    ]

    # 선택적 파라미터 추가
    if "conf_threshold" in exp_config:
        cmd.append(f"--conf_threshold={exp_config['conf_threshold']}")
    if "cutmix_prob" in exp_config:
        cmd.append(f"--cutmix_prob={exp_config['cutmix_prob']}")
    if "copy_paste_prob" in exp_config:
        cmd.append(f"--copy_paste_prob={exp_config['copy_paste_prob']}")
    if "aug_patience" in exp_config:
        cmd.append(f"--aug_patience={exp_config['aug_patience']}")
    if "aug_min_delta" in exp_config:
        cmd.append(f"--aug_min_delta={exp_config['aug_min_delta']}")
    
    # [핵심 수정] poly 옵션이 True면 --poly 플래그 추가
    if exp_config.get("poly", False):
        cmd.append("--poly")         # store_true 동작

    # EXP4_SEP: Stage별 파라미터 (필요시)
    if "stage1_patience" in exp_config:
        cmd.append(f"--stage1_patience={exp_config['stage1_patience']}")
    if "stage1_min_delta" in exp_config:
        cmd.append(f"--stage1_min_delta={exp_config['stage1_min_delta']}")
    if "stage2_patience" in exp_config:
        cmd.append(f"--stage2_patience={exp_config['stage2_patience']}")
    if "stage2_min_delta" in exp_config:
        cmd.append(f"--stage2_min_delta={exp_config['stage2_min_delta']}")

    return cmd

def run_experiments(experiments, base_path, data_path, python_exe, log_dir):
    """
    실험들을 순차/병렬로 실행
    """
    processes = []
    log_files = []

    print("=" * 60)
    print(f"Starting {len(experiments)} experiments")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    for exp in experiments:
        # 명령어 생성
        cmd = build_command(exp, base_path, data_path, python_exe)

        # 로그 파일 경로
        log_file = os.path.join(log_dir, f"{exp['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        log_files.append(log_file)

        print(f"\n[GPU {exp['gpu_id']}] Starting: {exp['name']}")
        print(f"  - Labeled data: {exp['labeled_num']} ({exp['labeled_num']/8.8:.0f}%)")
        print(f"  - Script: {exp['script']}")
        print(f"  - Poly Mode: {exp.get('poly', False)}")
        print(f"  - Classes: {exp.get('num_classes', 2)}")
        print(f"  - Log: {log_file}")

        # 프로세스 실행
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=base_path
            )
            processes.append({
                'process': process,
                'name': exp['name'],
                'gpu_id': exp['gpu_id'],
                'log_file': log_file
            })

    print("\n" + "=" * 60)
    print("All experiments started. Monitoring...")
    print("=" * 60)

    # 프로세스 모니터링
    monitor_processes(processes)

    return log_files

def monitor_processes(processes):
    """
    실행 중인 프로세스들을 모니터링
    """
    start_time = time.time()
    last_detailed_update = 0

    while True:
        # 모든 프로세스 상태 확인
        running = []
        completed = []

        for p_info in processes:
            if p_info['process'].poll() is None:
                running.append(p_info)
            else:
                completed.append(p_info)

        # 시간 표시
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        # 실시간 상태 수집
        status_info = []
        for p_info in processes:
            status_data = get_simple_status(p_info['log_file'])
            current_iter = status_data['current_iter']
            best_dice = status_data['best_dice']
            status = "RUN" if p_info['process'].poll() is None else "END"

            status_info.append(
                f"{p_info['name'][:15]:15s} [{status}] "
                f"Iter:{current_iter:5d} Best:{best_dice:.4f}"
            )

        # 한 줄로 상태 표시
        print(f"\r[{hours:02d}:{minutes:02d}] {' | '.join(status_info)}", end='', flush=True)

        # 모두 완료되면 종료
        if not running:
            print("\n\n" + "=" * 60)
            print("All experiments completed!")
            print("=" * 60)

            print("\nFinal Results:")
            for p_info in completed:
                return_code = p_info['process'].returncode
                status = "✓ SUCCESS" if return_code == 0 else f"✗ FAILED (code: {return_code})"

                status_data = get_simple_status(p_info['log_file'])
                final_iter = status_data['current_iter']
                best_dice = status_data['best_dice']

                print(f"\n[GPU {p_info['gpu_id']}] {p_info['name']}: {status}")
                print(f"  Final Iteration: {final_iter}")
                print(f"  Best Dice Score: {best_dice:.4f}")

            break

        # 120초마다 상세 상태 출력
        if elapsed - last_detailed_update >= 120:
            print(f"\n\n{'='*80}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status Update:")
            print(f"{'='*80}")

            for p_info in processes:
                status_data = get_simple_status(p_info['log_file'])
                status = "RUNNING" if p_info['process'].poll() is None else "COMPLETED"

                print(f"\n{p_info['name']} (GPU {p_info['gpu_id']}) - {status}")
                print(f"  Current Iteration: {status_data['current_iter']}")
                print(f"  Best Dice Score: {status_data['best_dice']:.4f}")

            print(f"{'='*80}\n")
            last_detailed_update = elapsed

        time.sleep(5)

def main():
    parser = argparse.ArgumentParser(description='AD-MT Parallel Training Script')
    parser.add_argument('--base_path', type=str, default='C:/ai-agent/repo/AD-MT',
                        help='Base path of AD-MT project')
    parser.add_argument('--data_path', type=str, default='C:/ai-agent/data/Kvasir',
                        help='Path to Kvasir dataset')
    parser.add_argument('--python_exe', type=str, default='C:/Users/sys73/.conda/envs/admt/python.exe',
                        help='Python executable path')
    parser.add_argument('--log_dir', type=str, default='C:/ai-agent/logs',
                        help='Directory for log files')
    
    # [수정] Poly 옵션 (터미널에서 입력 가능)
    parser.add_argument('--poly', action='store_true', default=True, help='Enable poly learning rate decay')
    
    parser.add_argument('--custom_experiments', action='store_true',
                        help='Use custom experiment configurations')

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    
    # [수정] 인자 전달
    experiments = create_experiment_configs(use_poly=args.poly)

    print("=" * 60)
    print("AD-MT Training Script (Final Version)")
    print("=" * 60)
    print(f"Base path: {args.base_path}")
    print(f"Data path: {args.data_path}")
    print(f"Python: {args.python_exe}")
    print(f"Experiments to run: {len(experiments)}")
    print(f"Poly Decay Mode: {'ON' if args.poly else 'OFF'}")

    print("\nExperiment configurations:")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']}")
        print(f"   - GPU: {exp['gpu_id']}")
        print(f"   - Script: {exp['script']}")
        print(f"   - Poly: {exp.get('poly', False)}")
        print(f"   - Classes: {exp.get('num_classes', 2)}")

    response = input("\nProceed with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return

    log_files = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"[Sequential Run] Experiment {i}/{len(experiments)}: {exp['name']}")
        single_log_files = run_experiments([exp], args.base_path, args.data_path, args.python_exe, args.log_dir)
        log_files.extend(single_log_files)
        print(f"[Sequential Run] Finished: {exp['name']}")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Log files created:")
    for log_file in log_files:
        print(f"  - {log_file}")

    print("\nTo monitor logs in real-time:")
    print(f"  Get-Content {args.log_dir}\\*.log -Wait (PowerShell)")

    print("\nTo test the models:")
    for exp in experiments:
        # 모델 저장 경로 추정 (Kvasir 데이터셋 구조에 맞춤)
        model_path = f"results/Kvasir/{exp['name']}_{exp['labeled_num']}_labeled/unet/unet_best_stu_model.pth"
        print(f"  python code/test_kvasir_2d.py --model_path=\"{model_path}\" --root_path=\"{args.data_path}\" --num_classes=2")

    print("\n" + "=" * 60)
    print("Script completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    if os.name == 'nt':
        default_python = "C:/Users/sys73/.conda/envs/admt/python.exe"
    else:
        default_python = "python"

    import sys
    # 기본 경로 설정이 안되어있을 경우를 대비한 안전장치
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--base_path', 'C:/ai-agent/repo/AD-MT',
            '--data_path', 'C:/ai-agent/data/Kvasir',
            '--python_exe', default_python,
            '--log_dir', 'C:/ai-agent/logs'
        ])

    main()