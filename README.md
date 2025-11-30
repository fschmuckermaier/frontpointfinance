# FrontPointFinance Monte Carlo Simulator

A Monte Carlo-based simulation tool to model portfolio development over time, supporting saving or withdrawing scenarios. Customize inputs like starting capital, yearly cashflows, asset allocation, returns, volatility, fees, inflation, and crash risk. Visualize investment outcomes interactively.

---

## Features

- Run thousands of simulation paths with adjustable parameters  
- Model stock and fixed income allocations with customizable return distributions  
- Account for inflation, fees (TER), dividends, taxes, and crash probabilities  
- Interactive plots showing portfolio value trajectories and distributions over time  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- Streamlit  
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/frontpointfinance.git
   cd frontpointfinance

2. Optional: Create environment:

python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt

### Run the app

streamlit run app.py

