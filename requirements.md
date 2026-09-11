# RL Auction Agent Visualization & Debugging Dashboard — Requirements

## 1. Purpose

Build a **local, browser-based visualization and debugging application** for a Python reinforcement-learning project.

The application is intended primarily as a **research/debugging tool for a master's thesis**, not as a production platform.

The main goal is to make it easy to answer questions such as:

- What is the RL agent doing at a particular timestep?
- Which vehicles are bidding?
- Which charging stations are being selected?
- What bids are being submitted?
- How does behavior relate to vehicle state such as SOC and destination?
- What happens to individual vehicles over time?
- Does the learned policy appear sensible?
- How does behavior change between epochs?
- Are strange reward improvements caused by sensible behavior or by unintended policy/environment interactions?
- Can a complete rollout be replayed without rerunning the simulation?

The visualization must therefore prioritize **fast interactive inspection and debugging over visual polish**.

---

# 2. Current RL/training architecture

The project is currently a Python project running on a workstation.

The simulator is based on **SimPy**, wrapped by a custom `GymWrapper` whose main purpose is to discretize simulation time for RL.

The training process:

1. Creates the environment and actor/critic.
2. Runs multiple rollout environments in parallel using `ProcessPoolExecutor`.
3. Each worker executes a rollout.
4. Worker rollout buffers are returned to the trainer.
5. Buffers are concatenated across workers.
6. GAE is computed.
7. PPO updates are performed.
8. Training continues for the next epoch.

The relevant trainer object contains:

- `self.env`
- `self.actor`
- `self.critic`
- `self.num_vehicles`
- `self.num_stations`
- `self.num_operators`
- configuration information
- parallel rollout count
- rollout length
- training epoch information

The dashboard should **not control the live SimPy simulation in the first implementation**.

Instead, the simulation/training process should record diagnostic information and the dashboard should load and replay recorded rollout/episode data.

---

# 3. Core architectural principle

Use a **Python backend + browser-based dashboard**.

Recommended initial technology:

- Python
- Plotly Dash
- Plotly for charts
- an H3-compatible map visualization
- pandas / NumPy for analysis
- PyTorch tensors may be converted to NumPy/pandas during recording
- Parquet or another simple serialized format for persistent diagnostic recordings

Do NOT introduce React, TypeScript, a separate REST API, WebSockets, databases, Docker, authentication, or cloud infrastructure unless later required.

The initial application should run locally:

```text
python dashboard/app.py
```

and expose a local Dash server such as:

```text
http://localhost:8050
```

The dashboard may eventually be reachable from another machine on the LAN, but this is not a requirement for the MVP.

---

# 4. Separation between simulation and dashboard

The dashboard must be decoupled from the SimPy simulation.

Preferred architecture:

```text
                 TRAINING
                    |
                    v
        +-------------------------+
        | SimPy + GymWrapper + RL |
        +------------+------------+
                     |
                     | diagnostic recording
                     v
        +-------------------------+
        |   Episode / Run Data    |
        +------------+------------+
                     |
                     v
        +-------------------------+
        |      Dash Dashboard     |
        +------------+------------+
                     |
                     v
                  Browser
```

The dashboard must not need to execute SimPy when the user changes the displayed timestep.

For example:

```text
User moves timestep:
143 -> 144 -> 145 -> 146
```

must only retrieve already-recorded data.

It must NOT do:

```text
run_simulation_until(144)
run_simulation_until(145)
...
```

The purpose of recording is to make replay effectively instantaneous.

---

# 5. Rollout recording

Create a diagnostic recorder that can be called from the training/rollout process.

The recorder should preserve enough information to reconstruct the visualization at every timestep.

The fundamental conceptual object is:

```text
Run
  Epoch
    Rollout / Environment
      Timestep
        Vehicle state
        Vehicle action
        Station state
        Reward
        Additional environment information
```

The system must support:

- multiple training epochs
- multiple parallel environments/rollouts per epoch
- multiple operators
- 336 timesteps per rollout by default
- approximately 150 vehicles
- approximately 70 real charging stations

The exact numbers must NOT be hard-coded in the dashboard.

They should be read from the recording/configuration.

---

# 6. Existing operator buffer structure

During an epoch, `combined_operator_buffers` is a list.

Each list entry corresponds to one operator.

Each operator entry is a dictionary with these keys:

```python
[
    {
        'allocations',
        'pricing',
        'log_probs',
        'rewards',
        'masks',
        'vehicles',
        'cs_distances',
        'user_distances',
        'stations_mask',
        'value'
    },
    ...
]
```

The dashboard requirements below distinguish between data already available in these buffers and additional simulation state that must be recorded.

---

# 7. Existing action data

## 7.1 Allocations

For each operator:

```text
allocations.shape =
(
    num_parallel_envs * steps_per_rollout,
    num_vehicles_in_fleet
)
```

Each row corresponds to one rollout timestep.

Each column corresponds to one vehicle in that operator's fleet.

The value is the selected charging-station index.

There are:

```text
0 ... 69
```

for real charging stations and:

```text
70
```

for the dummy/no-assignment station.

The dummy station therefore means:

> this vehicle was not assigned to a real charging station at this timestep.

The dashboard must understand this distinction.

### Required display behavior

Real station assignment:

```text
vehicle -> Station 23
```

Dummy assignment:

```text
vehicle -> No assignment
```

Do NOT display station index 70 as if it were a physical station.

The dummy station should not appear as a physical charging station on the map.

The value `70` should be treated as a configurable `NO_ASSIGNMENT` action.

The implementation should obtain the actual dummy index from configuration where possible rather than hard-coding `70`.

---

# 8. Existing pricing/bid data

For each operator:

```text
pricing.shape =
(
    num_parallel_envs * steps_per_rollout,
    num_vehicles_in_fleet
)
```

Each value corresponds to one vehicle.

If a vehicle selected a real charging station, `pricing` contains its bidding price.

If the vehicle selected the dummy station, the price is always zero.

The dashboard must therefore interpret:

```text
allocation != dummy_station
```

as an active station bid.

And:

```text
allocation == dummy_station
```

as no station bid.

Do not interpret a zero price alone as sufficient to determine whether a vehicle bid.

---

# 9. Existing reward data

For each operator:

```text
rewards.shape =
(
    num_parallel_envs * steps_per_rollout
)
```

Each entry is a scalar reward for the corresponding rollout timestep.

The dashboard must provide:

- total reward
- reward per timestep
- reward per epoch
- reward per operator
- reward distributions where useful

Do not assume that the scalar reward is directly attributable to one vehicle unless additional per-vehicle reward information is later recorded.

---

# 10. Existing value data

For each operator:

```text
value.shape =
(
    num_parallel_envs * steps_per_rollout
    [plus bootstrap timestep depending on buffer construction]
)
```

The value represents the critic's estimated state value.

The dashboard should initially support:

- value over timestep
- reward vs value
- optional return vs value
- value evolution during a rollout

Value visualization is secondary to the environment/action visualization.

---

# 11. Existing observation data

The operator buffer also contains:

```text
vehicles
cs_distances
user_distances
stations_mask
masks
```

and global observations contain station/user information.

These tensors should be made available to the recorder if useful.

However, the dashboard should NOT expose raw neural-network tensors by default.

Instead, convert them into meaningful semantic values wherever possible.

Raw tensor inspection can be added later as a developer/debug view.

---

# 12. Additional simulation state that should be recorded

The current RL buffers are not sufficient for the desired visualization.

The recorder should therefore additionally capture environment-level information at every timestep.

At minimum, vehicle state should contain:

```text
vehicle_id
operator_id

soc

last_known_position
destination

position_h3
destination_h3

reservation status
reservation start
reservation end

charging commitment status
committed station

current charging station, if applicable

whether vehicle placed a bid
selected station
bid price

whether vehicle selected dummy/no-assignment
```

Exact names may differ according to the existing simulator.

Do not rewrite the simulator's internal state model unnecessarily.

The recorder should adapt to the existing objects.

---

# 13. Position uncertainty

The simulator does not always know the exact current position of a vehicle.

Important rule:

- If last known position == destination, current position is known exactly.
- Otherwise, the vehicle is assumed to be traveling toward its destination and exact current position is uncertain.

This must be explicitly represented in the dashboard.

Recommended semantic fields:

```text
position_status:
    "known"
    "estimated"
```

or equivalent.

If an estimated current position is available from the simulator, record it.

If not, retain:

```text
last_known_position
destination
position_known = false
```

Do not falsely present an inferred position as ground truth.

---

# 14. H3 spatial representation

The city uses H3.

The dashboard's main spatial visualization should be an H3-based city map.

The map should show:

- H3 cells / city spatial context
- charging stations
- vehicles
- optionally destinations
- optionally station utilization
- optionally station prices
- optionally bid activity

The H3 resolution should come from configuration/data.

Do not hard-code one H3 resolution.

---

# 15. Main dashboard layout

The MVP should have one primary **Episode Replay / Debugging** page.

Suggested layout:

```text
+---------------------------------------------------------------+
| Epoch | Rollout | Timestep 143 / 336 | < > | PLAY | SPEED     |
+---------------------------------------------------------------+

+-----------------------------------+---------------------------+
|                                   | Fleet                     |
|                                   |                           |
|                                   | filters                   |
|                                   |                           |
|          H3 CITY MAP              | vehicle table             |
|                                   |                           |
|                                   |                           |
|                                   |                           |
+-----------------------------------+---------------------------+

+-----------------------------------+---------------------------+
| Selected vehicle / station        | Decision / state detail  |
| history                            |                           |
+-----------------------------------+---------------------------+
```

The precise styling is not important.

Information density and usability are more important.

---

# 16. Global controls

At the top of the dashboard:

## Epoch selector

Select the training epoch.

Example:

```text
Epoch: [ 17 v ]
```

## Rollout selector

Select which parallel environment/rollout to inspect.

Example:

```text
Rollout: [ 3 v ]
```

## Timestep selector

A slider:

```text
0 ------------------●---------------- 335
                     143
```

Show:

```text
Step 143 / 336
```

## Navigation controls

Provide:

- previous timestep
- next timestep
- play
- pause
- jump to beginning
- jump to end

## Playback speed

At minimum:

```text
0.5x
1x
2x
5x
```

Playback must operate on recorded data and therefore be fast.

---

# 17. Main H3 map

The H3 map is the central visualization.

At every timestep show:

## Charging stations

For each station:

- location
- station ID
- occupancy/utilization if available
- capacity
- current price if available
- number of bids if available

Station markers should be visually distinct from vehicles.

Clicking a station should select it.

---

# 18. Vehicle visualization

Each vehicle should appear at its relevant spatial location.

Vehicle visualization should communicate:

- vehicle identity on selection
- SOC
- position certainty
- bid/no bid
- station selected
- optionally urgency

Do not put 150 text labels permanently on the map.

Use markers and show details on hover/click.

Suggested marker semantics:

- known position: normal marker
- uncertain/estimated position: visually different marker, outline, opacity, or icon
- vehicle with active bid: optional highlighting
- selected vehicle: strong highlight

Do not overload color with too many meanings.

---

# 19. Destination visualization

Do not display destination lines for all 150 vehicles by default.

That will likely create an unreadable map.

Instead:

- show destination only for the selected vehicle
- optionally allow a "show destinations" layer
- optionally show destination markers when a vehicle is selected

For selected vehicle:

```text
Current / last-known position
          |
          | movement context
          v
Destination
```

If exact current position is uncertain, make this explicit.

---

# 20. Map layer controls

Provide toggles for layers such as:

```text
[x] Vehicles
[x] Charging stations
[ ] Destinations
[ ] Station occupancy
[ ] Station prices
[ ] Bids
[ ] Winning bids
[ ] Vehicle urgency
[ ] H3 cells
```

The exact available layers depend on recorded data.

This is important because the map should distinguish:

- environment state
- agent actions
- outcomes

For example, a busy station should not automatically be interpreted as a station the agent prefers.

---

# 21. Fleet panel

The fleet panel should be a **sortable/filterable table**, not 150 permanent cards.

Required columns:

```text
Vehicle ID
SOC
Position
Destination
Reservation
Bid
Selected station
Bid price
Commitment
Status
```

Exact names can be shortened for screen space.

---

# 22. Fleet filtering

Provide filters such as:

```text
SOC < 10%
SOC < 20%
SOC < 30%

Has reservation

Has charging commitment

Placed bid

No bid

Selected real station

Selected dummy/no-assignment
```

Additional useful filters:

```text
Selected station = ...
Operator = ...
Destination H3 = ...
Current H3 = ...
```

Filters should update the fleet table and ideally highlight matching vehicles on the map.

---

# 23. Vehicle selection

Clicking a vehicle in either:

- the map
- the fleet table

should select the vehicle globally.

Selection should update:

- map highlighting
- destination
- vehicle state panel
- action information
- historical charts
- station information if relevant

---

# 24. Selected vehicle detail panel

For the selected vehicle show:

```text
Vehicle 37

SOC
18%

Position
H3(...)

Position certainty
Estimated

Destination
H3(...)

Reservation
17:30–18:00

Charging commitment
None

Current action
Station 23

Bid
€0.36
```

If the allocation is the dummy station:

```text
Current action
No assignment

Bid
None
```

The panel should make the current decision immediately understandable.

---

# 25. Selected vehicle history

Show recent/full history for the selected vehicle.

At minimum:

## SOC over time

```text
SOC
 ^
 |\
 | \
 |  \____
 +----------> timestep
```

## Bid history

Show:

- bid price
- selected station
- no-bid periods

## Station commitment history

Show when the vehicle:

- had no commitment
- obtained a commitment
- changed commitment

## Reservation timeline

Show reservation intervals.

These views are particularly important for understanding temporal policy behavior.

---

# 26. Decision inspection

The dashboard should have a dedicated section explaining the selected vehicle's decision at the selected timestep.

Example:

```text
Decision at timestep 143

Vehicle state
-------------
SOC                 18%
Destination         H3(...)
Reservation          Yes
Reservation time     17:30
Commitment           None

Agent action
------------
Selected station     23
Bid price            0.36

Environment
-----------
Station 23 capacity  4
Station 23 occupancy 2
Station 23 price     0.31

Other candidate stations
------------------------
Station 12   distance ...   price ...   occupancy ...
Station 23   distance ...   price ...   occupancy ...
Station 41   distance ...   price ...   occupancy ...
```

The exact candidate information depends on available environment data.

This section should answer:

> "What was the agent seeing and what did it choose?"

Do not claim that the panel explains the neural network's internal reasoning. It shows the observed state and resulting action.

---

# 27. Station selection

Clicking a charging station should show:

```text
Station ID
Location
Capacity
Current occupancy
Current price

Number of bids
Number of winning bids

Vehicles bidding for this station
```

The fleet table should optionally filter to vehicles bidding on the selected station.

This creates a useful workflow:

```text
Station
   ↓
Which vehicles bid here?
   ↓
What were their SOCs?
   ↓
What bids did they submit?
```

---

# 28. Station history

For a selected station show:

- occupancy over time
- price over time
- number of bids over time
- winning bids if available
- involved vehicles

This can help identify congestion and auction dynamics.

---

# 29. Reward/training view

Add a separate page or collapsible section for training metrics.

At minimum:

## Reward by epoch

Plot:

```text
epoch -> total reward
```

## Reward by operator

Plot one line per operator.

## Reward by timestep

For a selected rollout:

```text
timestep -> reward
```

## Value by timestep

```text
timestep -> critic value
```

Potential later metrics:

- mean bid
- number of active bids
- number of no-assignment decisions
- station utilization
- successful allocations
- reservation satisfaction
- charging commitments
- entropy
- policy loss
- critic loss

Only add metrics when their semantics are reliable.

---

# 30. Strategy analysis page

After the MVP, add a strategy analysis page.

Useful plots include:

## Bid vs SOC

Scatter plot:

```text
SOC -> bid price
```

## Bid vs urgency

If urgency is available:

```text
urgency -> bid price
```

## Bid vs station price

```text
station price -> agent bid
```

## Bid vs distance

```text
distance -> bid
```

## SOC vs assignment probability

```text
SOC -> probability of choosing real station
```

## Station selection distribution

```text
station ID -> number of assignments
```

## No-assignment frequency

```text
epoch -> percentage of dummy/no-assignment actions
```

These plots are intended to answer:

> "Has the policy learned a plausible strategy?"

---

# 31. Epoch comparison

Later, support comparing two epochs or two runs.

Example:

```text
Epoch A: 2
Epoch B: 20
Timestep: 143
Rollout: 0
```

Show side-by-side:

```text
H3 map A       H3 map B
Fleet A        Fleet B
Metrics A      Metrics B
```

This is particularly useful for observing learning.

Ideally the comparison should use fixed evaluation scenarios/seeds where possible so that differences are attributable to policy changes rather than environment randomness.

---

# 32. Run comparison

The storage design should allow multiple training runs.

A run should have metadata such as:

```text
run_id
timestamp
configuration
number of epochs
model/checkpoint identifier
random seed information
```

Possible structure:

```text
runs/
    run_001/
        metadata.json
        training_metrics.parquet
        episodes/
            ...
    run_002/
        metadata.json
        training_metrics.parquet
        episodes/
            ...
```

The dashboard should eventually be able to select a run.

This is useful when changing:

- reward functions
- PPO parameters
- network architecture
- environment configuration

---

# 33. Evaluation episodes

Training rollouts and diagnostic/evaluation rollouts should be distinguishable.

Prefer a workflow where:

```text
training
    |
    | every N epochs
    v
fixed evaluation scenarios
    |
    v
record complete diagnostic episodes
```

This provides stable episodes for comparing policy versions.

The dashboard should support evaluation recordings exactly like normal recordings.

---

# 34. Data model

The implementation should create a clean internal data model rather than passing raw tensors throughout the UI.

Recommended conceptual structures:

```python
Run
    run_id
    metadata
    epochs

Epoch
    epoch_id
    rollouts
    training_metrics

Rollout
    rollout_id
    timesteps

Timestep
    timestep
    vehicles
    stations
    reward
    operator_metrics

VehicleState
    vehicle_id
    operator_id
    soc
    position_h3
    last_known_position_h3
    destination_h3
    position_known
    reservation
    commitment
    allocation
    bid_price

StationState
    station_id
    h3
    capacity
    occupancy
    price
    bids
```

Use dataclasses, TypedDicts, or another simple typed representation.

Avoid unnecessary framework complexity.

---

# 35. Tensor conversion

The existing data uses PyTorch tensors.

The dashboard should not directly manipulate CUDA tensors.

At the recording boundary:

```python
tensor.detach().cpu().numpy()
```

or equivalent should be used.

The dashboard should work with CPU-side serialized data.

Never transfer tensors from GPU to browser.

---

# 36. Parallel environment handling

The existing buffers concatenate multiple parallel environments along the first dimension.

The recorder must preserve the distinction between:

```text
parallel environment / rollout ID
```

and:

```text
timestep
```

Do NOT permanently lose this information by only storing:

```text
flat_index
```

Instead store:

```text
rollout_id
timestep
```

explicitly.

If:

```text
num_parallel_envs = 10
steps_per_rollout = 336
```

then the dashboard should be able to reconstruct:

```text
rollout 0, timestep 0
...
rollout 0, timestep 335

rollout 1, timestep 0
...
```

etc.

---

# 37. Operator handling

The project is multi-operator.

The dashboard must preserve:

```text
operator_id
```

for vehicle/action information.

Provide an operator filter where useful.

The map may show all operators together, but selected operator views should be possible.

---

# 38. Performance requirements

The dashboard should feel instantaneous during replay.

Target:

- timestep change: effectively immediate
- vehicle selection: effectively immediate
- filter changes: preferably < 200 ms for normal datasets
- map updates: preferably < 500 ms
- playback: smooth enough for 1x and 2x replay

Do not rerun simulation for visualization.

Precompute derived data where useful.

For example:

```text
is_bid
is_dummy_assignment
selected_station
```

should ideally be derived once during recording/loading rather than recalculated repeatedly in every UI callback.

---

# 39. Storage

Start simple.

A practical MVP can use:

- Parquet for tabular vehicle/timestep data
- Parquet for station/timestep data
- JSON for run metadata/configuration

Example:

```text
run_001/
    metadata.json

    training_metrics.parquet

    vehicle_states.parquet
    station_states.parquet
```

Possible vehicle table columns:

```text
run_id
epoch
rollout_id
timestep
operator_id
vehicle_id

soc
position_h3
destination_h3
position_known

has_reservation
reservation_start
reservation_end

has_commitment
committed_station

allocation
is_dummy_assignment
bid_price
is_bid
```

Possible station table:

```text
run_id
epoch
rollout_id
timestep
station_id
h3

capacity
occupancy
price

num_bids
...
```

Do not prematurely normalize into a relational database.

---

# 40. Raw data retention

The recorder should optionally support a diagnostic level.

For example:

```python
recording_level="minimal"
recording_level="full"
```

Minimal could contain:

- reward
- action
- SOC
- basic state

Full could additionally contain:

- observation tensors
- distances
- masks
- candidate station information
- critic value
- other simulator state

This allows normal training to remain lightweight while enabling detailed debugging runs.

---

# 41. Do not store unnecessary PyTorch training internals

The dashboard does NOT initially need:

- model weights
- gradients
- optimizer state
- log probabilities
- hidden activations
- raw logits

These can be added later if there is a specific debugging need.

`log_probs` are explicitly not important for the initial visualization.

---

# 42. Handling the dummy station

This is a critical semantic requirement.

The dashboard must represent:

```text
allocation == dummy station
```

as:

```text
NO ASSIGNMENT
```

and:

```text
allocation != dummy station
```

as:

```text
REAL STATION ASSIGNMENT
```

Pricing is:

```text
dummy -> 0
real station -> actual bid
```

Never interpret:

```text
pricing == 0
```

alone as the no-bid indicator.

The allocation determines whether the vehicle submitted a real station action.

---

# 43. Suggested status semantics

A vehicle can have several independent statuses.

Do not collapse all of them into one ambiguous field.

For example:

```text
Reservation:       yes/no
Commitment:        yes/no
Bid:               yes/no
Assignment:        station/no assignment
Position certainty: known/estimated
```

This avoids misleading UI logic.

---

# 44. Environment state vs agent behavior

The dashboard must distinguish between:

## Environment state

Examples:

- SOC
- destination
- reservation
- station occupancy
- station price
- station capacity
- distances

and:

## Agent behavior

Examples:

- selected station
- bid price
- no-assignment action

and:

## Outcome

Examples:

- winning bid
- successful allocation
- charging commitment
- reward

This distinction is important for debugging.

A station being busy does not mean the agent caused the station to be busy.

---

# 45. "Why did this happen?" workflow

The most important debugging workflow should be:

```text
1. Select timestep
2. Find unusual vehicle/station
3. Select vehicle
4. Inspect state
5. Inspect action
6. Inspect candidate stations
7. Inspect recent history
8. Inspect outcome/reward
```

The UI should make this workflow easy.

---

# 46. Example debugging scenario

The dashboard should make the following investigation possible:

> "At timestep 143, why did vehicle 37 bid €0.36 for station 23?"

The user should be able to select vehicle 37 and see:

```text
Vehicle 37
-------------------------
SOC:              18%
Destination:      H3(...)
Reservation:      yes
Commitment:       none

Action:
Station:          23
Bid:              €0.36

Candidate stations:
Station 12
Station 23
Station 41
...

Recent history:
SOC declining
No commitment
Previous bids ...
```

This is much more valuable than simply showing a reward curve.

---

# 47. Reward debugging

Because reward modeling is still under development, the dashboard should make reward behavior inspectable.

Provide:

```text
reward by epoch
reward by timestep
reward distribution
reward by operator
```

If reward components exist later, support:

```text
total reward
    =
    component A
    + component B
    + component C
```

Do not assume reward components exist now.

The architecture should allow them to be added.

---

# 48. Future reward decomposition

If the reward is later changed to include multiple terms, the recorder should support:

```text
reward_total
reward_component_1
reward_component_2
...
```

The dashboard can then show:

```text
Total reward
┌───────────────────────┐
│                       │
└───────────────────────┘

Reward components
A ──────────────────────
B ───────────────
C ───────
```

This will be particularly useful during reward engineering.

---

# 49. UX principles

The application should be:

- functional
- information-dense
- fast
- easy to navigate
- visually clear

Avoid:

- excessive animation
- decorative dashboards
- giant KPI cards
- unnecessary gradients
- excessive color coding
- 3D visualizations
- complex navigation

The user is a researcher/debugger, not an executive dashboard consumer.

---

# 50. Recommended MVP implementation order

Implement in this order:

## Phase 1 — Recording

1. Create diagnostic recorder.
2. Preserve epoch.
3. Preserve rollout ID.
4. Preserve timestep.
5. Record vehicle state.
6. Record allocation.
7. Record pricing.
8. Record station state.
9. Record reward.
10. Serialize recording.

## Phase 2 — Basic dashboard

1. Start Dash application.
2. Load one recording.
3. Add timestep slider.
4. Add H3 map.
5. Add vehicles.
6. Add stations.
7. Add fleet table.
8. Add selected vehicle panel.

## Phase 3 — Replay

1. Previous/next timestep.
2. Play/pause.
3. Playback speed.
4. Vehicle selection.
5. Station selection.

## Phase 4 — Debugging

1. Vehicle filters.
2. Selected vehicle history.
3. Bid history.
4. Reservation/commitment timeline.
5. Station detail.
6. Reward timeline.

## Phase 5 — Analysis

1. Strategy plots.
2. Epoch comparison.
3. Run comparison.
4. Evaluation episode comparison.

Do not implement Phase 4/5 before Phase 1–3 are usable.

---

# 51. MVP acceptance criteria

The first version is successful if all of the following work:

### Replay

The user can select:

```text
epoch
rollout
timestep
```

and see the corresponding simulation state without running SimPy.

### Map

The H3 city map shows:

- vehicles
- charging stations

### Vehicle

The user can select a vehicle and see:

- SOC
- position
- destination
- position certainty
- reservation
- commitment
- allocation
- bid

### Fleet

The user can filter/sort vehicles.

### Station

The user can select a station and see relevant station information and bidding vehicles.

### Actions

The dashboard correctly distinguishes:

```text
real station assignment
```

from:

```text
dummy/no assignment
```

### Replay speed

Changing timestep is fast enough that the user can scrub through an episode comfortably.

### Training

The user can see reward by timestep and epoch.

If all of these work, the dashboard is already useful enough for the thesis.

---

# 52. Non-goals for the first implementation

Do NOT implement initially:

- live simulation control
- remote simulation execution
- REST API
- authentication
- cloud hosting
- multi-user support
- database server
- React frontend
- model explainability algorithms
- SHAP
- neural activation visualization
- distributed dashboard infrastructure
- automatic natural-language explanations of decisions

These may be future extensions but are not required.

---

# 53. Recommended project structure

Suggested structure:

```text
project/
│
├── simulation/
│   └── existing simulation code
│
├── simulation/auction_agent/
│   └── existing RL code
│
├── diagnostics/
│   ├── __init__.py
│   ├── recorder.py
│   ├── models.py
│   └── loader.py
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   ├── layout.py
│   ├── callbacks.py
│   ├── map_view.py
│   ├── fleet_view.py
│   ├── vehicle_view.py
│   ├── station_view.py
│   └── training_view.py
│
└── runs/
    └── ...
```

Adapt this to the existing repository rather than restructuring the entire project.

---

# 54. Code quality requirements

The implementation should:

- use type hints
- avoid global mutable state where possible
- separate data loading from UI callbacks
- separate visualization construction from callback logic
- use small functions
- document assumptions about tensor dimensions
- validate recording dimensions
- fail clearly if required data is missing

The dashboard must not modify simulation state.

---

# 55. Important implementation detail: tensor indexing

The existing buffers concatenate parallel rollout data.

The coding agent must carefully establish the mapping:

```text
flat index
    ->
rollout_id
timestep
```

If:

```text
steps_per_rollout = T
```

then for a simple concatenation:

```python
rollout_id = flat_index // T
timestep = flat_index % T
```

However, this mapping must be confirmed against the actual worker rollout buffer construction before implementation.

Do not blindly assume this if the worker stores an additional batch dimension.

The coding agent should inspect the actual `run_worker_rollout` implementation and tensor shapes before implementing the recorder.

---

# 56. Important implementation detail: current state vs observation

The dashboard needs semantic simulator state, not merely the RL observation tensor.

The coding agent should inspect:

- `GymWrapper`
- `RolloutRunner`
- `run_worker_rollout`
- vehicle/user classes
- charging station classes
- reservation/commitment objects

to identify the authoritative source of:

- SOC
- last-known position
- destination
- reservation
- charging commitment
- station occupancy
- station price
- vehicle IDs
- station IDs

Do not reconstruct these values from neural-network observations if the simulator already has authoritative state.

---

# 57. Important implementation detail: do not break training

The diagnostic recorder must have minimal impact on training performance.

Recommended approach:

```text
training normally
    |
    +--> diagnostic recording enabled only when requested
```

For example:

```python
diagnostic_recording = True
```

or a configuration option.

Normal large-scale training should not necessarily serialize every timestep of every worker.

A good approach is:

- train normally
- periodically run one or a few evaluation/diagnostic episodes
- record those in full

---

# 58. Configuration

The dashboard should not hard-code:

- number of vehicles
- number of stations
- number of operators
- rollout length
- dummy station index
- H3 resolution

These should come from:

- recording metadata
- configuration
- derived dataset metadata

The recording should contain enough metadata for the dashboard to interpret itself.

---

# 59. Development strategy

Build against **one saved rollout first**.

Do not start by supporting every training epoch, every worker, and every historical run.

The first milestone should be:

```text
one recorded rollout
+
336 timesteps
+
150 vehicles
+
70 stations
+
interactive map
+
fleet table
```

Once this works, generalize to multiple epochs/runs.

This keeps implementation risk low.

---

# 60. Final design philosophy

The dashboard is a **debugger for learned behavior**.

It should make the following transition easy:

```text
"I see reward improving."
              |
              v
"What is the agent actually doing?"
              |
              v
"Which vehicles/stations are responsible?"
              |
              v
"What state did the agent have?"
              |
              v
"What action did it choose?"
              |
              v
"What happened afterwards?"
              |
              v
"Is this behavior desirable?"
```

The dashboard should prioritize this workflow above all other features.

The first implementation should be intentionally small and should be possible to build in several days rather than becoming a months-long frontend/software-engineering project.
