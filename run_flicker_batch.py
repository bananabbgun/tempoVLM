import os
import subprocess
import argparse

# 測試設定
# 格式: "編號": ["Frame代碼1", "Frame代碼2", ...]
# 邏輯: 000500 -> Frame 5 (除以 100)
SCENE_CONFIGS = {
    "0757": [500, 1000, 1200, 1700, 2900, 4400, 5200, 7100],
    "0736": [200, 1400, 1700, 2900, 6400, 7000],
    "0747": [700, 1000, 3300, 5000],
    "0761": [1200, 1900, 2400, 3300, 4700],
    "0784": [700, 1500, 2300, 2700, 3100, 3800],
    "0712": [900, 1200, 1600, 2600, 4000],
    "0739": [800, 2000, 2600, 3700],
    "0768": [200, 700, 1300, 1900, 3600],
    "0785": [600, 1600, 3400],
    "0721": [1100, 1800, 2100, 3000, 3300],
    "0776": [500, 1600]
}

def get_frame_str(raw_list):
    """將 000500 轉為 '5' 並組合成字串 '5, 10, ...'"""
    frames = [str(int(x / 100)) for x in raw_list]
    return ", ".join(frames)

def main():
    parser = argparse.ArgumentParser()
    # 預設模型路徑 (已確認位於 models/checkpoints_gru_occ 下)
    default_model = "./models/checkpoints_gru_occ/gru_checkpoint_epoch6.pt"
    parser.add_argument("--model_path", type=str, default=default_model, help="Path to model checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    # 檢查模型路徑是否存在 (僅在非 dry_run 時檢查)
    if not args.dry_run and not os.path.exists(os.path.dirname(args.model_path)) and "gru" in args.model_path:
        print(f"[Warning] Model directory {os.path.dirname(args.model_path)} not found locally.")
        print("Please ensure the path is correct on the server.")

    print(f"====== Starting Batch Test ({len(SCENE_CONFIGS)} Scenes) ======")
    
    # 設定 YOLO Config 路徑到 /tmp2 以避免 Disk Quota Exceeded
    env = os.environ.copy()
    env["YOLO_CONFIG_DIR"] = "/tmp2/b12902026/yolo_config"
    if not os.path.exists(env["YOLO_CONFIG_DIR"]):
        os.makedirs(env["YOLO_CONFIG_DIR"], exist_ok=True)
            
    for scene_id, raw_frames in SCENE_CONFIGS.items():
        scene_name = f"scene{scene_id}_00"
        data_root = f"./scannet_data/scannet_frames_test/{scene_name}"
        occlusion_frames = get_frame_str(raw_frames)
        
        cmd = [
            "python", "complete_demo.py",
            "--model_path", args.model_path,
            "--data_root", data_root,
            "--demos", "occlusion",
            "--occlusion_type", "yolo_all", # 雖然是 v6.1 Flickering，但保留使用者習慣的參數
            "--occlusion_frames", occlusion_frames,
            "--max_scenes", "1"
        ]
        
        print(f"\n[Running] {scene_name} with Occlusion: [{occlusion_frames}]")
        
        if args.dry_run:
            print("Command:", " ".join(cmd))
        else:
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to run {scene_name}: {e}")
                
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
