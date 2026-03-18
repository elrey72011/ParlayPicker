export PYTHONPATH=$PYTHONPATH:$(pwd)
export ODDS_API_KEY="test"
python -m streamlit run streamlit_app.py --server.headless true --server.port 8501 > streamlit.log 2>&1 &
echo $! > streamlit.pid
