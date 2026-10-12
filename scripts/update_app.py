import sys

with open("app.py", "r") as f:
    lines = f.readlines()

new_content = """            st.session_state.history = [initial_state]
            st.session_state.sim_id = initial_state.simulation_id
            st.session_state.sim_active = True
            st.session_state.sim_finished = False
            
            import threading
            def run_sim_thread(state):
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(orchestrator.run_full_simulation(state))
                except Exception as e:
                    import traceback
                    logger.log(f"Simulation Error: {e}", "ERROR", "App")
                    SwarmIO.save_live_state(state.simulation_id, {
                        "status": "error",
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    })
                    
            t = threading.Thread(target=run_sim_thread, args=(initial_state,), daemon=True)
            try:
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                add_script_run_ctx(t)
            except ImportError:
                pass
            t.start()
            
    if st.session_state.get("sim_active", False):
        st.divider()
        st.subheader("📡 Real-time Agent Comms & Physics")
        
        @st.fragment(run_every="1s")
        def sim_status_fragment():
            sim_id = st.session_state.sim_id
            live_state = SwarmIO.load_live_state(sim_id)
            if not live_state:
                st.info("Initializing...")
                return
            
            status = live_state.get("status", "running")
            iteration = live_state.get("iteration", 0)
            max_iter = live_state.get("max_iterations", 1)
            msg = live_state.get("message", "Running...")
            
            st.text(msg)
            st.progress(iteration / max_iter if max_iter > 0 else 0)
            
            snap_data = live_state.get("latest_snapshot")
            if snap_data:
                live_data = []
                for aid, agent in snap_data.get("agent_states", {}).items():
                    cfg = snap_data.get("agent_configs", {}).get(agent.get("config_id"))
                    if not cfg: continue
                    pos = agent.get('position', {'x':0, 'y':0, 'z':0})
                    live_data.append({
                        "Agent": cfg.get("name", "Unknown Agent"),
                        "Position": f"({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}, {pos.get('z', 0):.1f})",
                        "Thought": agent.get("latest_thought", "...")
                    })
                    
                for item in live_data:
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.markdown(f"**{item['Agent']}**")
                        st.caption(f"Pos: {item['Position']}")
                    with col_b:
                        st.info(item['Thought'])
                        
            if status == "complete":
                st.session_state.history = []
                for snap_dict in live_state.get("history", []):
                    try:
                        snap = SwarmSnapshot.model_validate(snap_dict)
                        st.session_state.history.append(snap)
                    except Exception as e:
                        pass
                
                st.success(f"Simulation Complete! Data exported to: {live_state.get('export_path')}")
                st.session_state.sim_active = False
                st.session_state.sim_finished = True
                st.session_state.sim_final_state = live_state
                st.rerun()
            elif status == "error":
                st.error(f"Error during simulation: {live_state.get('message')}")
                st.code(live_state.get('traceback', ''))
                st.session_state.sim_active = False
                st.rerun()
                
        sim_status_fragment()

    if st.session_state.get("sim_finished", False):
        live_state = st.session_state.get("sim_final_state", {})
"""

new_lines = [line + "\n" for line in new_content.split('\n')]
# Adjust lines 698 to 797 inclusive
# python list is 0-indexed, so 697 to 797
result_lines = lines[:697] + new_lines + lines[797:]

with open("app.py", "w") as f:
    f.writelines(result_lines)
