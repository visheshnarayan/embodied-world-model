"""Local experiment dashboard: streamlit run scripts/dashboard.py"""
from pathlib import Path
import json
import subprocess
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"artifacts"; FIG=ART/"figures"
st.set_page_config(page_title="Embodied World Model",layout="wide")
st.title("Embodied World Model RL")
st.caption("NVIDIA SingleArm data → JAX world model → learned simulator → planning/policy evaluation")

with st.sidebar:
    st.header("Project status")
    st.success("State/action benchmark available")
    st.info("Visual preview is available; visual prediction and physical-robot control are future stages.")
    st.markdown("**Current methods**\n\n- World-model prediction\n- Behavior cloning\n- CEM planning\n- Learned-dynamics Gymnasium env")

tab_data,tab_visual,tab_model,tab_control,tab_files=st.tabs(["Dataset","Visual data","World model","Control","Artifacts"])
with tab_data:
    st.subheader("SingleArm dataset")
    data_root=ROOT/"data/single_arm/panda-stack-wide"; files=list(data_root.glob("data/**/*.parquet")) if data_root.exists() else []
    c1,c2,c3=st.columns(3); c1.metric("Downloaded episodes",len(files)); c2.metric("Task","panda-stack-wide"); c3.metric("Videos loaded","No")
    st.write("The current benchmark uses Parquet state/action data only. Each episode contains Panda robot state, object poses, and 7-D end-effector/gripper actions.")
    if files:
        sample=pd.read_parquet(files[0]); st.write("Sample schema"); st.dataframe(sample.head(10),use_container_width=True)
    else: st.warning("Download data/single_arm first.")
with tab_visual:
    st.subheader("SingleArm visual observations")
    videos=sorted((ROOT/"data/single_arm/panda-stack-wide/videos").glob("**/*.mp4"))
    if videos:
        labels=[str(v.relative_to(ROOT)) for v in videos]
        choice=st.selectbox("Episode/camera",range(len(videos)),format_func=lambda i: labels[i],key="video_choice")
        selected=videos[choice]
        @st.cache_data(show_spinner=False)
        def read_video(path): return Path(path).read_bytes()
        clip=read_video(str(selected))
        st.video(clip,format="video/mp4",autoplay=False,loop=True,muted=False)
        st.download_button("Download selected clip",clip,file_name=selected.name,mime="video/mp4")
        probe=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames,r_frame_rate","-of","default=noprint_wrappers=1",str(selected)],capture_output=True,text=True,check=False)
        frame_count=1
        for line in probe.stdout.splitlines():
            if line.startswith("nb_read_frames="):
                try: frame_count=int(line.split("=",1)[1])
                except ValueError: pass
        frame_index=st.slider("Inspect frame",0,max(0,frame_count-1),min(frame_count//2,frame_count-1),key="frame_index")
        frame=subprocess.run(["ffmpeg","-loglevel","error","-i",str(selected),"-vf",f"select=eq(n\\,{frame_index})","-frames:v","1","-f","image2pipe","-vcodec","png","pipe:1"],capture_output=True,check=False).stdout
        if frame: st.image(frame,caption=f"Frame {frame_index+1}/{frame_count}",width="stretch")
        st.caption(f"{frame_count} frames at 30 FPS ({frame_count/30:.2f} seconds). The downloaded samples are short task episodes, not blank clips.")
    else:
        st.warning("No sample videos found. Download visual samples with scripts/download_single_arm.py or the commands in the README.")
with tab_model:
    st.subheader("Prediction quality")
    scaling=ART/"single_arm_scaling.csv"; rollouts=ART/"rollout_error_100ep.csv"
    if scaling.exists():
        df=pd.read_csv(scaling); st.line_chart(df,x="episodes",y="val_mse",color="variant")
        st.dataframe(df,use_container_width=True)
    if rollouts.exists():
        st.write("Compounding rollout error"); st.line_chart(pd.read_csv(rollouts).set_index("horizon")["mse"])
    st.markdown("**Interpretation:** one-step error measures local physics prediction; multi-step error measures whether imagined futures remain useful for planning.")
with tab_control:
    st.subheader("Policy/planner comparison")
    st.write("These are evaluations inside the learned dynamics simulator, not physical robot trials.")
    st.dataframe(pd.DataFrame({"Controller":["Random","Behavior cloning","CEM"],"Mean dense return":[-0.0914,-0.0561,-0.0277],"Sparse success":[0.,0.,0.]}),use_container_width=True)
    image=FIG/"single_arm_scaling.png"
    if image.exists(): st.image(str(image),caption="Data scaling and preprocessing variants")
    st.warning("Current benchmark has a planning signal but no reliable sparse task success yet.")
with tab_files:
    st.subheader("Generated artifacts")
    for path in sorted(ART.rglob("*")):
        if path.is_file() and ".cache" not in str(path): st.write(str(path.relative_to(ROOT)))
    st.markdown("**Run more experiments from the terminal, then refresh this page.**")
