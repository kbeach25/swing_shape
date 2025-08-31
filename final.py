from pybaseball import statcast, playerid_reverse_lookup
from datetime import date
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

st.title("Swing Shape Analysis Tool")

@st.cache_data
# Get data from opening day until today, only pull and process data once
def getData():
    #raw_df = statcast('2025-03-27', str(date.today()))
    raw_df = statcast('2025-03-27', '2025-06-30')
    
    ids = raw_df['batter'].dropna().unique()
    id_df = playerid_reverse_lookup(ids, key_type='mlbam')

    id_to_name = {
    row['key_mlbam']: f"{row['name_first']} {row['name_last']}"
    for _, row in id_df.iterrows()
    }

    raw_df['batter_name'] = raw_df['batter'].map(id_to_name)
    
    return raw_df

# Statcast data from statcast
df = getData()

# Bat tracking data from csv
tracking_df = pd.read_csv("tracking_data.csv")

def intro_module():
    st.title("Introduction")

    st.markdown("""
    Welcome to the Swing Shape Analysis Tool. Here, you can use
    data gathered by bat sensors to explore how swing shape and
    mechanics relate to performance metrics.
    """
    )

    st.markdown("First, here is some of the terminology that will be used.")

    st.markdown("**Bat Speed: Speed (MPH) of the bat at impact.**")
    st.image("bs.png", width=400)
    st.markdown("<small>Via MLB.com</small>", unsafe_allow_html=True)

    st.markdown("**Time to Contact (TTC): Time (s) between heel strike and contact.**")
    st.image("ttc.gif", width=400)
    st.markdown("<small>MakeAGIF.com</small>", unsafe_allow_html=True)

    st.markdown("**Vertical Bat Angle: Angle (deg) of the bat at contact, where completely horizontal is 0.**")
    st.image("freddie.png", width=400)
    st.markdown("<small>Via MLB.com</small>", unsafe_allow_html=True)

    st.markdown("**Attack Angle: Angle (deg) perpendicular to the bat line at contact.**")
    st.image("fernando.png", width=400)
    st.markdown("<small>Via East Village Times</small>", unsafe_allow_html=True)

    st.markdown("**Swing Length: Arc length (ft) of a swing.**")
    st.image("swing_length.png", width=400)
    st.markdown("<small>Via MLB.com</small>", unsafe_allow_html=True)

    st.markdown("**Batting Average: A batter's rate of getting hits. Above 0.300 is generally considered elite.**")
    st.markdown("**On-Base-Percentage: A batter's rate of getting on base. Above 0.360 is generally considered elite.**")
    st.markdown("**Slugging Percentage: A batter's rate of total bases per at-bat. Above 0.500 is generally considered elite.**")
    st.markdown("**K%: A batter's rate of striking out.**")
    st.markdown("**BB%: A batter's rate of walking.**")

# Expected performance module function, using csv data
# Use RF for SLG, Linear regression for K%
def expected_performance(df):
    st.title("Expected Performance")

    df['vertical_bat_angle'] = df['vertical_swing_path'] * -1
    df['ttc'] = df.apply(lambda row: calculate_time_to_contact(row['avg_swing_speed'], row['avg_swing_length']), axis=1)

    # Copied from model_selection.py ############
    
    df_filtered = df.dropna(subset=[
    'vertical_bat_angle', 'attack_angle', 'ttc', 'avg_swing_speed',
    'batting_avg', 'slg_percent', 'on_base_percent', 'k_percent', 'bb_percent'
    ])

    X = df_filtered[['attack_angle', 'ttc', 'avg_swing_speed', 'vertical_bat_angle']]
    y = df_filtered[['batting_avg', 'slg_percent', 'on_base_percent', 'k_percent', 'bb_percent']]
    target_names = y.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # need performance scores for each target
    def get_metrics_per_target(y_true, y_pred, model_name, formula):
        metrics = []
        for i, target in enumerate(target_names):
            metrics.append({
                'model': model_name,
                'target': target,
                'formula': formula,
                'R2': r2_score(y_true.iloc[:, i], y_pred[:, i]),
                'MAE': mean_absolute_error(y_true.iloc[:, i], y_pred[:, i]),
                'RMSE': np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
            })
        return metrics

    results = []
    
    # End of code copied from model_selection.py ###########

    # Linear regression
    lr = MultiOutputRegressor(LinearRegression())
    lr.fit(X_train, y_train)

    rf = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    rf.fit(X_train, y_train)

    st.header("Enter Swing Data")
    st.markdown("**Batting Average, On-Base Percentage, and Walk (BB) Rate are too reliant on swing tendency data to be accurately predicted using swing shape data alone.**")

    col1, col2 = st.columns(2)

    with col1:
        attack_angle = st.number_input("Attack Angle", value=10.0)
        avg_swing_speed = st.number_input("Bat Speed", value=70.0)
    with col2:
        vertical_bat_angle = st.number_input("Vertical Bat Angle", value=-30.0)
        ttc = st.number_input("Time to Contact", value=0.14)

    if st.button("Predict Performance"):
        input_data = pd.DataFrame([{
            'attack_angle': attack_angle,
            'ttc': ttc,
            'avg_swing_speed': avg_swing_speed,
            'vertical_bat_angle': vertical_bat_angle
        }])

        # Make predictions
        rf_pred = rf.predict(input_data)
        lr_pred = lr.predict(input_data)

        predicted_slg = rf_pred[0][1]  # slg_percent is index 1
        predicted_k = lr_pred[0][3]    # k_percent is index 3

        # Max velocity formula: 1.2 * batspeed + 0.2 * pitch speed, assume 95
        max_ev = input_data['avg_swing_speed'].iloc[0] * 1.2 + 0.2 * 95

        st.subheader("Predicted Performance")
        st.metric(label="SLG", value=f"{predicted_slg:.3f}")
        st.metric(label="K", value=f"{predicted_k:.3f}")
        st.metric(label="Max Exit Velocity", value=f'{max_ev:.3f}')


    # Additional plots
    st.subheader("Swing Shape vs Performance")

    feature_options = ['avg_swing_speed', 'vertical_bat_angle', 'ttc', 'attack_angle']
    target_options = ['batting_avg', 'slg_percent', 'on_base_percent', 'k_percent', 'bb_percent']

    col3, col4 = st.columns(2)
    with col3:
        x_axis = st.selectbox("Select X-axis (Swing Metric)", feature_options, index=0)
    with col4:
        y_axis = st.selectbox("Select Y-axis (Performance Metric)", target_options, index=1)

    fig = px.scatter(
        df_filtered, 
        x=x_axis, 
        y=y_axis, 
        trendline="ols", 
        title=f"{x_axis} vs {y_axis}",
        labels={x_axis: x_axis.replace("_", " ").title(), y_axis: y_axis.replace("_", " ").title()},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    
# Cluster plots and performance breakdown module (NOTE: Working with csv here, not statcast data. Averages are given)
def performance_data(df):
    st.title("Swing Classification")

    # Add time to contact to df
    df['ttc'] = df.apply(lambda row: calculate_time_to_contact(row['avg_swing_speed'], row['avg_swing_length']), axis=1)

    # Vertical swing path is positive, but need it to be negative
    df['vertical_bat_angle'] = df['vertical_swing_path'] * -1

    # Use bat tracking features
    features = ['avg_swing_speed', 'attack_angle', 'vertical_bat_angle', 'ttc']

    # There shouldn't be any missing data but just in case
    X = df[features].dropna()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=4, random_state=42) 
    gmm.fit(X_scaled)

    # Predict cluster labels and append to DataFrame
    df['GMM_cluster'] = gmm.predict(scaler.transform(df[features]))
    df['GMM_cluster'] = df['GMM_cluster'].astype(str)

    cluster_order = ['0', '1', '2', '3']
    color_map = {'0': '🔴', '1': '🟢', '2': '🔵', '3': '🟠'}
    color_sequence = ['red', 'green', 'blue', 'orange']

    plot_options = [
        'k_percent', 'bb_percent', 'batting_avg', 'slg_percent', 
        'on_base_percent', 'avg_swing_speed', 'avg_swing_length', 
        'attack_angle', 'vertical_bat_angle', 'ttc'
    ]
    x_axis = st.selectbox("Select X-axis variable", plot_options, index=plot_options.index('avg_swing_speed'))
    y_axis = st.selectbox("Select Y-axis variable", plot_options, index=plot_options.index('attack_angle'))

    fig = px.scatter(
    df, 
    x=x_axis, 
    y=y_axis,
    color='GMM_cluster',
    category_orders={'GMM_cluster': cluster_order},
    title='GMM Clustering',
    labels={'GMM_cluster': 'Cluster'},
    color_discrete_sequence=color_sequence
    )

    st.plotly_chart(fig)

    st.markdown("Clusters are formed based on Bat Speed, Time to Contact (TTC), Vertical Bat Angle, and Attack Angle.")

    df['cluster_label'] = df['GMM_cluster'].apply(lambda x: f"{color_map.get(x, '')}")
    #df.to_csv("clusters.csv")
    
    st.subheader("Cluster Swing Data")
    avg_cols = ['avg_swing_speed', 'ttc', 'vertical_bat_angle', 'attack_angle']
    cluster_avg = df.groupby('GMM_cluster')[avg_cols].mean().round(3)
    cluster_avg = cluster_avg.reset_index()
    cluster_avg['Cluster'] = cluster_avg['GMM_cluster'].map(lambda x: f"{color_map.get(x)}")
    cluster_avg = cluster_avg.drop(columns='GMM_cluster')
    cluster_avg = cluster_avg[['Cluster'] + avg_cols]
    st.dataframe(cluster_avg)

    st.subheader("Cluster Performance Data")
    avg_cols2 = ['batting_avg', 'slg_percent', 'on_base_percent', 'k_percent', 'bb_percent']
    cluster_avg2 = df.groupby('GMM_cluster')[avg_cols2].mean().round(3)
    cluster_avg2 = cluster_avg2.reset_index()
    cluster_avg2['Cluster'] = cluster_avg2['GMM_cluster'].map(lambda x: f"{color_map.get(x)}")
    cluster_avg2 = cluster_avg2.drop(columns='GMM_cluster')
    cluster_avg2 = cluster_avg2[['Cluster'] + avg_cols2]
    st.dataframe(cluster_avg2)

    st.subheader("Notable Members of each Cluster:")
    st.markdown("**🔴: Vladimir Guerrero Jr. (R) | Juan Soto (L)**")
    st.markdown("**🟢: Mookie Betts (R) | Christian Yelich (L)**")
    st.markdown("**🔵: Paul Goldschmidt (R) | Cody Bellinger (L)**")
    st.markdown("**🟠: Mike Trout (R) | James Wood (L)**")

    st.subheader("Enter Swing Metrics Below")

    col1, col2 = st.columns(2)
    with col1:
        swing_speed = st.number_input("Bat Speed", value=70.0)
        vba = st.number_input("Vertical Bat Angle", value=-30.0)
    with col2:
        attack_angle = st.number_input("Attack Angle", value=10.0)
        ttc = st.number_input("Time to Contact", value=0.13)

    user_data = pd.DataFrame([{
        'avg_swing_speed': swing_speed,
        'attack_angle': attack_angle,
        'vertical_bat_angle': vba,
        'ttc': ttc
    }])
    
    input_scaled = scaler.transform(user_data)
    probs = gmm.predict_proba(input_scaled)[0]
    percentages = [f"{p * 100:.1f}%" for p in probs]

    # Display probabilities
    st.markdown("**Probability represents how well your swing profile matches each grouping.**")
    prob_df = pd.DataFrame({
        'Cluster': [f"{color_map[str(i)]}" for i in range(4)],
        'Probability': percentages
    })
    st.dataframe(prob_df)

    
def calculate_time_to_contact(bat_speed, swing_length):
    # time = 2 * distance / (initial velocity + final velocity)

    ft_per_sec = bat_speed * 5280 / 3600
    ttc = round(2 * swing_length / (ft_per_sec), 3)
    return ttc

# Draw heatmap
def plot_heatmap(df, hand):
    
    fig, ax = plt.subplots(facecolor='none')
    ax.set_facecolor('none')

    # Filter for switch hitters
    df = df[df['stand'] == hand]

    # Only want pitches that result in hits to be visible
    df = df[df['events'].isin(['double', 'triple', 'home_run'])]
    
    # There should always be something here but just in case
    if not df.empty:
        sz_top = df['sz_top'].iloc[0]
        sz_bot = df['sz_bot'].iloc[0]
        ax.add_patch(plt.Rectangle((-0.83, sz_bot), 1.66, sz_top - sz_bot,
                     edgecolor='white', facecolor='none', linewidth=2))

        x = df['plate_x']
        y = df['plate_z']
        ax.scatter(x, y, color='red', s=100)

        # Adjust plot style
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 5)
        ax.set_title(f"Extra Base Hits (Catcher POV)", color='white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.patch.set_alpha(0.0) 
        plt.tight_layout()

        st.pyplot(fig)
        

# Plot 3d swing figure
def plot_swing(attack_angle, vba, avg_swing_length, sz_top, sz_bot, handedness):
    # VBA is relative to x axis, where 0 is horizontal and -90 is vertical
    # Attack Angle is relative to y axis, where 0 is vertical and 90 is horizontal
    # Swing length determines arc length

    if handedness == "R":
        side = 1
    else:
        side = -1

    # Use 33.5 inch bat
    bat_length = 33.5 * side
    sweet_spot = 29.5 * side # estimate

    # given degrees, need radians
    attack_rad = np.radians(attack_angle)
    vba_rad = np.radians(vba)

    # Convert strike zone measurements to inches, width is always 17
    zone_top = sz_top * 12
    zone_bot = sz_bot * 12

    fig = go.Figure()

    # hide axis labels
    fig.update_layout(
    scene=dict(
        xaxis=dict(
            visible=False,
            showgrid=False,
            showticklabels=False,
            title=''
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            showticklabels=False,
            title=''
        ),
        zaxis=dict(
            visible=False,
            showgrid=False,
            showticklabels=False,
            title=''
        )
    ),
    margin=dict(l=0, r=0, b=0, t=30)
    )

    # Draw plate
    fig.add_trace(go.Scatter3d(
        x=[-8.5, 8.5], y=[17, 17], z=[0, 0],
        mode='lines',
        line=dict(color='white', width=2),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[-8.5, -8.5], y=[17, 8.5], z=[0, 0],
        mode='lines',
        line=dict(color='white', width=2),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[8.5, 8.5], y=[17, 8.5], z=[0, 0],
        mode='lines',
        line=dict(color='white', width=2),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[-8.5, 0], y=[8.5, 0], z=[0, 0],
        mode='lines',
        line=dict(color='white', width=2),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[8.5, 0], y=[8.5, 0], z=[0, 0],
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))

    # Fill in plate with white
    fig.add_trace(go.Mesh3d(
        x=[-8.5, 8.5, 8.5, 0, -8.5],  
        y=[17,   17,  8.5, 0,  8.5], 
        z=[0,    0,   0,   0,  0],   
        i=[0, 0, 2],
        j=[1, 2, 3],
        k=[2, 4, 4],
        color='white',
        opacity=1.0,
        showscale=False
    ))

    # Draw strike zone
    fig.add_trace(go.Mesh3d(
        x=[-8.5, 8.5, 8.5, -8.5],    
        y=[17,   17,   17,   17],    
        z=[zone_bot, zone_bot, zone_top, zone_top], 
        i=[0], j=[1], k=[2],        
        color='rgba(200, 200, 200, 0.4)', 
        opacity=0.6,
        name='Strike Zone',
        showscale=False
    ))
    fig.add_trace(go.Mesh3d(
        x=[-8.5, 8.5, -8.5, 8.5],
        y=[17,   17,   17,   17],
        z=[zone_bot, zone_bot, zone_top, zone_top],
        i=[0], j=[2], k=[3],  
        color='rgba(200, 200, 200, 0.4)',
        opacity=0.6,
        showscale=False,
        name='Strike Zone'
    ))

    # Draw bat, this stuff is subject to change
    
    # calculate directional components
    x_dir = np.cos(vba_rad)
    y_dir = np.sin(attack_rad) * side
    z_dir = np.sin(vba_rad) * side

    # normal vector
    norm = np.sqrt(x_dir**2 + y_dir**2 + z_dir**2)
    x_dir /= norm
    y_dir /= norm
    z_dir /= norm

    # Start point of bat
    x_start = -25.125 * side
    y_start = 17
    z_start = zone_top

    # End point, or approximate contact point
    x_end = (x_start + bat_length * x_dir)
    y_end = (y_start + bat_length * y_dir)
    z_end = (z_start + bat_length * z_dir)

    # Draw bat
    fig.add_trace(go.Scatter3d(
        x=[x_start, x_end],
        y=[y_start, y_end],
        z=[z_start, z_end],
        mode='lines',
        line=dict(color='lightblue', width=20),
        name='Bat',
        showlegend=False
    ))

    # Convert swing length from ft to inches
    swing_len = avg_swing_length * 12

    # Batter height is about top of zone * 1.72
    batter_height = 1.79 * zone_top

    # estimate hand starting position to be about a foot below the top of the batter's head
    hand_start = batter_height

    # Swing shape start point, sweet spot of contact point
    shape_x_start = x_start + sweet_spot * x_dir
    shape_y_start = y_start + sweet_spot * y_dir
    shape_z_start = z_start + sweet_spot * z_dir

    # Swing shape end point
    shape_x_end = (-23*side - bat_length * x_dir) # Inside, outside, avg distance off plate is 28, but bat will be a little closer
    shape_y_end = 17 - swing_len*(2/3) # Catcher to pitcher
    shape_z_end = hand_start + sweet_spot * z_dir  # Up and down

    # Swing shape middle point
    swing_x_middle = 6*x_dir#(-25*side - bat_length * x_dir)
    swing_y_middle = 17 - swing_len
    swing_z_middle = hand_start + sweet_spot*side - batter_height*1.2

    P0 = np.array([shape_x_start, shape_y_start, shape_z_start])
    P1 = np.array([swing_x_middle, swing_y_middle, swing_z_middle])
    P2 = np.array([shape_x_end, shape_y_end, shape_z_end])

    # Interpolate points along the Bézier curve
    t_vals = np.linspace(0, 1, 50)  # Increase for smoother curve
    bezier_points = [(1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2 for t in t_vals]
    bezier_points = np.array(bezier_points)

    # Add the curved swing path to the plot
    fig.add_trace(go.Scatter3d(
        x=bezier_points[:, 0],
        y=bezier_points[:, 1],
        z=bezier_points[:, 2],
        mode='lines',
        line=dict(color='blue', width=8),
        name='Swing Shape',
        showlegend=False
    ))


    return fig
    
    

# manually calculate avg/obp/slg from statcast data
def calculate_slash(df):
    desc = df['events'].dropna().astype(str).str.lower().str.strip()

    # Any non empty events cell means something happened, so an official plate appearance was recorded
    plate_appearances = desc[desc != ""].count()

    # At bats exclude walk, hbp, and sacrifices
    at_bats_mask = ~desc.isin(['walk', 'hit_by_pitch']) & ~desc.str.contains('sac') & (desc != "")
    at_bats = at_bats_mask.sum()

    singles = (desc == "single").sum()
    doubles = (desc == "double").sum()
    triples = (desc == "triple").sum()
    homers = (desc == "home_run").sum()
    walks = (desc == "walk").sum()
    hbp = (desc == "hit_by_pitch").sum()
    sac_bunt = (desc == "sac_bunt").sum()
    strikeouts = (desc == "strikeout").sum()
    
    hits = singles + doubles + triples + homers
    avg = round(hits / at_bats, 3)

    on_base = hits + walks + hbp
    obp = round(on_base / (plate_appearances - sac_bunt), 3)

    tb = singles * 1 + doubles * 2 + triples * 3 + homers * 4
    slg = round(tb / at_bats, 3)

    k_rate = round(strikeouts * 100 / plate_appearances, 1)
    walk_rate = round(walks * 100 / plate_appearances, 1)

    return f'AVG: {avg} | OBP: {obp} | SLG: {slg} | K%: {k_rate}% | BB%: {walk_rate}%'
    
    
def bat_tracking(df):
    # Attack Angle, VBA, Swing Length, Bat Speed
    avg_attack_angle = round(pd.to_numeric(df['attack_angle'], errors='coerce').mean(), 1)
    avg_vba = round(pd.to_numeric(df['swing_path_tilt'], errors='coerce').mean(), 1) * -1 # Should be a negative number
    avg_swing_length = round(pd.to_numeric(df['swing_length'], errors='coerce').mean(), 1)
    #avg_bat_speed = round(pd.to_numeric(df['bat_speed'], errors='coerce').mean(), 1)
    avg_bat_speed = round(pd.to_numeric(df['bat_speed'], errors='coerce')[pd.to_numeric(df['bat_speed'], errors='coerce') > 60].mean(), 1)

    return avg_attack_angle, avg_vba, avg_swing_length, avg_bat_speed
    
    

# Function for swing visualizer module
def swing_visual_mod(df):
    st.title("Swing Visualizer")

    # Dropdown menu to select batter
    batter = st.selectbox("Choose a Batter", df['batter_name'].dropna().unique())
    batter_df = df[df['batter_name'] == batter]

    # Statcast defined top and bottom of strike zone 
    sz_top = pd.to_numeric(batter_df['sz_top'], errors='coerce').mean()
    sz_bot = pd.to_numeric(batter_df['sz_bot'], errors='coerce').mean()

    # Return a string to be displayed
    slash = calculate_slash(batter_df)
    avg_attack_angle, avg_vba, avg_swing_length, avg_bat_speed = bat_tracking(batter_df)
    ttc = calculate_time_to_contact(avg_bat_speed, avg_swing_length)

    st.markdown(f'**{slash}**')
    st.markdown(f'**Bat Speed: {avg_bat_speed} mph**')
    st.markdown(f'**Time to Contact: {ttc} sec**')
    st.markdown(f'**Attack Angle: {avg_attack_angle} degrees**')
    st.markdown(f'**Vertical Bat Angle: {avg_vba} degree**')
    st.markdown(f'**Swing Length: {avg_swing_length} ft**')

    batter_side = batter_df['stand'].dropna().unique().tolist()

    # Handle switch hitters
    if len(batter_side) > 1:
        handed_choice = st.radio("Choose side to view from:", ["Right-Handed (R)", "Left-Handed (L)"])
    
        if handed_choice == "Right-Handed (R)":
            handedness = "R"
        elif handed_choice == "Left-Handed (L)":
            handedness = "L"

    else:
         handedness = batter_side[0]   

    if handedness:
        fig = plot_swing(avg_attack_angle, avg_vba, avg_swing_length, sz_top, sz_bot, handedness)
        st.plotly_chart(fig, use_container_width=True)
        plot_heatmap(batter_df, handedness)

    

def main():
    #st.title("Swing Shape Analysis Tool")

    st.sidebar.title("Select a Module")
    module = st.sidebar.radio("Modules:", ["Introduction", "Swing Visualizer", "Swing Classification", "Expected Performance"])

    if module == "Introduction":
        intro_module()
        
    elif module == "Swing Visualizer":
        swing_visual_mod(df)
        
    elif module == "Swing Classification":
        performance_data(tracking_df)
        
    elif module == "Expected Performance":
        expected_performance(tracking_df)

main()


