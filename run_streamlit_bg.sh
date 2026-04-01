#!/bin/bash
streamlit run main.py --server.port 3000 --server.address 0.0.0.0 > streamlit_output.log 2>&1 &
echo $! > streamlit.pid
sleep 5
