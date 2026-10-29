# DNA Inheritance Simulator

An interactive Streamlit app based on the recombination simulation in `DL_Poject_7th_sem.ipynb`. It compares motif-based fitness scores for the two input sequences, a naive child, and an optimized simulated child.

This is an educational simulation, not a trained model or a clinical/biological prediction tool.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Enter parent DNA sequences containing only A, T, G, and C. The simulation processes complete 20-base genes and reports when trailing bases are omitted. Results can be downloaded as FASTA and CSV files.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) and choose **Create app**.
3. Select `Himanshu-1950/DNA_Prediction`, branch `main`, and entrypoint `app.py`.
4. Deploy. The dependencies are listed in `requirements.txt`.