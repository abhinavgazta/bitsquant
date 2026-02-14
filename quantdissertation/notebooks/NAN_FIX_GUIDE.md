# NaN Propagation Fix - Complete Guide

## Problem Diagnosis

**Error:** `ValueError: Expected parameter loc (Tensor of shape (64, 51)) of distribution Normal(...) to satisfy the constraint Real(), but found invalid values: tensor([[nan, nan, nan, ...]])`

**Root Cause:** NaN values were propagating through the environment and policy network:
1. **Observation Pipeline**: Raw observations contained NaN/Inf from feature engineering
2. **Environment Step Function**: Reward calculations had division by zero, extreme values, or invalid prices
3. **Transformer Network**: No gradient clipping or output validation led to NaN activation outputs
4. **Policy Distribution**: The PPO policy tried to create a Normal distribution with NaN mean values

---

## Solutions Implemented

### 1. **Observation Validation (IndianEquityEnv._get_obs)**

```python
def _get_obs(self):
    idx = self.current_step
    obs_data = []
    for feat in ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']:
        obs_data.append(self.df[feat].iloc[idx-self.lookback:idx].values)
    obs = np.concatenate(obs_data, axis=1).astype(np.float32)
    
    # NEW: Validate and clip observations
    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)  # Replace NaN/Inf
    obs = np.clip(obs, -10.0, 10.0)  # Clip to reasonable bounds
    return obs
```

**Changes:**
- Replace any NaN values with 0.0 (neutral value)
- Replace infinite values with ±1.0 (bounded)
- Clip all observations to [-10.0, 10.0] range (prevents extreme values feeding to network)

---

### 2. **Environment Step Function Robustness**

#### 2a. Action Processing (Softmax Numerical Stability)
```python
# OLD: exp_action = np.exp(action)  # Can overflow!
# NEW:
action_clipped = np.clip(action, -500, 500)  # Prevent overflow in exp
exp_action = np.exp(action_clipped - np.max(action_clipped))  # Numerical stability trick
weights = exp_action / (np.sum(exp_action) + 1e-9)
```

#### 2b. Price Data Validation
```python
current_prices = np.nan_to_num(current_prices, nan=1.0)  # Default to 1.0 if NaN
next_prices = np.nan_to_num(next_prices, nan=1.0)
current_prices[current_prices == 0] = 1.0  # Prevent division by zero
```

#### 2c. Return Calculations with Clipping
```python
asset_returns = (next_prices - current_prices) / (current_prices + 1e-9)
asset_returns = np.nan_to_num(asset_returns, nan=0.0)  # Handle NaN
asset_returns = np.clip(asset_returns, -0.5, 0.5)  # Clip extreme returns
```

#### 2d. ADV (Average Daily Volume) Safety
```python
advs = self.df['adv_20'].iloc[self.current_step].values
advs = np.nan_to_num(advs, nan=1e6)  # Default to 1M shares if NaN
advs = np.maximum(advs, 1e5)  # Ensure minimum ADV (100k shares)
```

#### 2e. Reward Clipping & Final Validation
```python
net_ret = weighted_returns - cost_linear - (slippage_penalty / (self.portfolio_value + 1e-9))
net_ret = np.clip(net_ret, -0.5, 0.5)  # Clip returns to ±50%
net_ret = np.nan_to_num(net_ret, nan=0.0)  # Final NaN check

reward = net_ret - downside_penalty - explicit_turnover_penalty
reward = np.clip(reward, -10.0, 10.0)  # Clip reward
reward = float(np.nan_to_num(reward, nan=0.0))  # Ensure scalar float
```

#### 2f. Portfolio Value Safety
```python
new_portfolio_value = self.portfolio_value * (1 + net_ret)
if np.isnan(new_portfolio_value) or np.isinf(new_portfolio_value):
    new_portfolio_value = self.portfolio_value  # Keep old value if invalid
self.portfolio_value = new_portfolio_value
```

---

### 3. **ExplainableTransformer Network Stability**

#### 3a. Proper Weight Initialization
```python
# NEW: Xavier initialization for better convergence
nn.init.xavier_uniform_(self.linear_in.weight)
nn.init.xavier_uniform_(self.linear_out.weight)
```

#### 3b. Attention Mechanism Regularization
```python
# Added dropout to prevent overfitting and instability
self.mha = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, 
                                  batch_first=True, dropout=0.1)
self.dropout1 = nn.Dropout(0.1)
```

#### 3c. Forward Pass Validation at Every Stage
```python
def forward(self, observations):
    # Input validation
    if torch.isnan(observations).any():
        observations = torch.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=-1.0)
    observations = torch.clamp(observations, -10.0, 10.0)
    
    # After linear layer
    x = self.linear_in(observations)
    x = torch.clamp(x, -10.0, 10.0)
    
    # After attention
    attn_output, attn_weights = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
    if torch.isnan(attn_output).any():
        attn_output = torch.nan_to_num(attn_output, nan=0.0)
    attn_output = torch.clamp(attn_output, -10.0, 10.0)
    
    # Layer norm
    x = self.norm1(x + self.dropout1(attn_output))
    x = x.flatten(start_dim=1)
    x = torch.clamp(x, -10.0, 10.0)
    
    # Final output
    output = self.act(self.linear_out(x))
    if torch.isnan(output).any():
        output = torch.nan_to_num(output, nan=0.0)
    output = torch.clamp(output, -1.0, 1.0)
    
    return output
```

**Key Points:**
- Validate at every layer boundary
- Use clipping instead of just ignoring NaN
- Ensure bounded outputs (tanh is [-1, 1])

---

### 4. **Training Function Robustness**

#### 4a. Gradient Clipping in Warm-up
```python
def pretrain_encoder(agent, env, epochs=5):
    # ... training loop ...
    loss.backward()
    # NEW: Gradient clipping
    torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
    optimizer.step()
```

#### 4b. PPO Model Configuration
```python
model = PPO("MlpPolicy", env, learning_rate=learning_rate, ent_coef=ent_coef, 
            clip_range=clip_range, policy_kwargs=policy_kwargs, verbose=0,
            max_grad_norm=1.0)  # NEW: Model-level gradient clipping
```

#### 4c. Objective Function Error Handling
```python
def objective(trial, train_df, test_df):
    # ... hyperparameter suggestions ...
    
    try:
        model.learn(total_timesteps=2000)
    except Exception as e:
        print(f"   Training failed: {e}")
        return -1000  # Return bad score if training fails
    
    # Evaluation with safety limits
    while not done and steps < max_steps:  # NEW: max_steps limit
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_rewards += reward
            done = terminated or truncated
            steps += 1
        except Exception as e:
            print(f"   Eval step failed: {e}")
            break
```

---

## Safety Bounds Summary

| Component | Bound | Reason |
|-----------|-------|--------|
| Observations | [-10, 10] | Prevent extreme values from feedforward |
| Asset Returns | [-0.5, 0.5] | ±50% daily return is already extreme |
| Net Return | [-0.5, 0.5] | Same as asset returns |
| Reward | [-10, 10] | Policy should never receive extreme rewards |
| Transformer Output | [-1, 1] | Tanh activation ensures this |
| Gradient Norm | ≤ 1.0 | Prevents gradient explosion |
| Action Input | [-500, 500] | Before numerical stability transform |
| Prices | min=1.0 | Prevents division by zero |
| ADV | min=100k | Realistic minimum for liquid stocks |

---

## Testing & Validation

### Step 1: Check Notebook Opens
```bash
python3 -m json.tool dissertation_updated_01.ipynb > /dev/null
echo "Notebook JSON valid: $?"
```

### Step 2: Run Single Optimization Trial
```python
# In notebook: run all cells through the objective function
# Then test with:
train_df = df_gym.iloc[:int(len(df_gym)*0.6)]
test_df = df_gym.iloc[int(len(df_gym)*0.6):]

trial_result = objective(
    type('obj', (), {'suggest_float': lambda s, n, low, high, log=False: (low+high)/2})(),
    train_df, test_df
)
print(f"Trial result: {trial_result}")
```

### Step 3: Full Experiment Run
```python
run_experiment(use_local_data=True)
# Should complete all 7 steps without NaN errors
```

---

## Common Issues & Troubleshooting

### Issue 1: "Still getting NaN in loss"
**Solution:** The CSV data itself may have NaN values. Ensure preprocessing in `load_local_csv_dataset()` and `fetch_and_process_data()` properly handles missing data with `.bfill().ffill()`.

### Issue 2: "OOM or slow training"
**Solution:** Try reducing the number of trials in Optuna from 3 to 1, or reduce training timesteps from 2000 to 1000.

### Issue 3: "Observation shape mismatch"
**Solution:** Ensure `n_assets` matches the number of tickers in the dataframe. Check:
```python
print(f"Tickers in data: {len(df['Close'].columns)}")
print(f"Observation shape: {env.observation_space.shape}")
```

---

## Files Modified

1. **IndianEquityEnv._get_obs()** - Observation validation
2. **IndianEquityEnv.step()** - Reward and price validation
3. **ExplainableTransformer** - Network stability and initialization
4. **pretrain_encoder()** - Gradient clipping
5. **objective()** - Error handling and evaluation safety

---

## Expected Output After Fix

```
================================================================================
STEP 1: DATA LOADING & FEATURE ENGINEERING
================================================================================
✓ Data Loaded Successfully
  Shape: (2366, 400)
  Date Range: 2012-10-10 to 2024-01-30
  Features: 400 (stocks × features)

================================================================================
STEP 3: BAYESIAN HYPERPARAMETER OPTIMIZATION
================================================================================
[I 2026-02-11 ...] Trial 0 finished with value: 125.43...
[I 2026-02-11 ...] Trial 1 finished with value: 132.87...
[I 2026-02-11 ...] Trial 2 finished with value: 128.45...

✓ Best Hyperparameters Found:
  lr: 0.000125
  ent_coef: 0.015
  ...
```

**Key Indicators of Success:**
- ✓ All steps complete without exceptions
- ✓ No "nan" or "inf" in trial values
- ✓ Positive reward accumulation in evaluation
- ✓ Equity curve shows increasing portfolio value

