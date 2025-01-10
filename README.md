# distribution
Code allowing to shape your own distribution law. 

## README.md

# Interactive Token Unlock Schedule with Market Depth Provisioning

This Streamlit app simulates and visualizes token unlock schedules with dynamic market depth provisioning. It includes features for modeling token price behavior, vesting schedules, bear market effects, and liquidity provisioning. 

## Features

- **Tokenomics Inputs**: Define maximum supply, initial token price, and offset for simulation start.
- **Price Modeling**: Choose between a constant price or a stochastic price model using the Black-Scholes model.
- **Bear Market Simulation**: Configure bear market periods with increased sell pressure.
- **Custom Vesting Schedules**: Editable vesting parameters for various token categories.
- **Dynamic Market Depth**: Adjust market depth and liquidity provisioning over time.
- **Rewards Allocation**: Allocate rewards using a logistic distribution.
- **Visualization**: Interactive plots displaying unlock schedules, market depth, overflow, and ROI.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Configuration

- **Tokenomics Inputs**: Set maximum supply, initial price, and offset month.
- **Price Model**: Select constant or stochastic pricing and adjust parameters (mu, sigma).
- **Bear Market Periods**: Define bear market ranges and sell pressure.
- **Vesting Schedule**: Customize unlock percentages, lock-up periods, and sell pressure triggers.
- **Liquidity Provisioning**: Define additional liquidity at specific months.

## Visualization

- **Unlock Schedule**: Stacked bar chart of token unlocks by category.
- **Overflow Analysis**: Identify periods where market depth is exceeded.
- **ROI and Bear Market Impact**: Track ROI changes and highlight bear markets.

## License

Creative Commons Legal Code License
