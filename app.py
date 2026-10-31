import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title='Pro Car Price Prediction Dashboard',
    page_icon='🚗',
    layout='wide',
)

# Custom Background Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
  try:
    with open('car_price_model.pkl', 'rb') as f:
      model, feature_names, encoding_maps, r2, mae, feature_imp = pickle.load(f)
    return model, feature_names, encoding_maps, r2, mae, feature_imp
  except Exception as e:
    return None, None, None, None, None, None


model, feature_names, encoding_maps, r2, mae, feature_imp = load_artifacts()


@st.cache_data
def load_data():
  if os.path.exists('car_data.csv'):
    return pd.read_csv('car_data.csv')
  return None


df_raw = load_data()

st.title('🚗 Pro Car Price Prediction & AI Valuation Dashboard')
st.markdown(
    'Advanced Indian Car Market Intelligence with Hotness Meter & AI Assistant'
)

if model is None or df_raw is None:
  st.error('Model or Dataset not found! Please run model_building.py first.')
else:
  tab1, tab2, tab3, tab4, tab5 = st.tabs([
      '🚀 Price Prediction & Hotness',
      '⚖️ Comparative Analysis',
      '💰 EMI & Depreciation Planner',
      '📊 Model Insights',
      '🤖 AI Car Expert Chatbot',
  ])

  with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
      st.subheader('📝 Enter Car Specifications')

      brands = (
          sorted(df_raw['brand'].dropna().unique().tolist())
          if 'brand' in df_raw.columns
          else []
      )
      selected_brand = st.selectbox('Car Brand', brands)

      if 'brand' in df_raw.columns and 'model' in df_raw.columns:
        filtered_models = sorted(
            df_raw[df_raw['brand'] == selected_brand]['model']
            .dropna()
            .unique()
            .tolist()
        )
        if not filtered_models:
          filtered_models = sorted(df_raw['model'].dropna().unique().tolist())
      else:
        filtered_models = ['Default Model']

      selected_model = st.selectbox('Car Model', filtered_models)

      cities = (
          sorted(df_raw['city'].dropna().unique().tolist())
          if 'city' in df_raw.columns
          else ['Kolkata', 'Delhi', 'Mumbai']
      )
      selected_city = st.selectbox('City / Registration Location', cities)

      milage = st.number_input(
          'Fuel Mileage (km/l)',
          min_value=5.0,
          max_value=40.0,
          value=16.0,
          step=0.5,
      )
      car_age = st.slider(
          'Car Age (Years)', min_value=0, max_value=20, value=3
      )

      fuel_types = (
          sorted(df_raw['fuel_type'].dropna().unique().tolist())
          if 'fuel_type' in df_raw.columns
          else ['Petrol', 'Diesel']
      )
      transmissions = (
          sorted(df_raw['transmission'].dropna().unique().tolist())
          if 'transmission' in df_raw.columns
          else ['Manual', 'Automatic']
      )
      accidents = (
          sorted(df_raw['accident'].dropna().unique().tolist())
          if 'accident' in df_raw.columns
          else ['None', 'At least 1 accident']
      )

      selected_fuel = st.selectbox('Fuel Type', fuel_types)
      selected_trans = st.selectbox('Transmission', transmissions)
      selected_accident = st.selectbox('Accident History', accidents)

      predict_btn = st.button('🚀 Predict Car Price', use_container_width=True)

    with col2:
      st.subheader('🎨 Graphic Car Visual & Card')
      st.markdown(
          f"""
        <div style="padding: 25px; background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%); border-radius: 16px; text-align: center; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); margin-bottom: 20px; border: 2px solid #e2e8f0;">
            <div style="font-size: 50px; margin-bottom: 5px;">🚙</div>
            <h2 style="color: #1f2937; margin: 5px 0; font-size: 22px; font-weight: 800; text-transform: uppercase;">{selected_brand}</h2>
            <h1 style="color: #2563eb; margin: 5px 0; font-size: 28px; font-weight: bold;">{selected_model}</h1>
            <p style="color: #4b5563; font-weight: 600; font-size: 14px; margin: 5px 0;">📍 Location: {selected_city}</p>
            <hr style="border: 0; height: 1px; background: #cbd5e1; margin: 15px 0;">
            <div style="display: flex; justify-content: space-around; color: #374151; font-size: 13px; font-weight: bold;">
                <span>⛽ {selected_fuel}</span>
                <span>⚙️ {selected_trans}</span>
                <span>📅 {car_age} Yrs</span>
            </div>
        </div>
        """,
          unsafe_allow_html=True,
      )

      hotness_score = (
          88
          if selected_brand.lower()
          in ['maruti', 'hyundai', 'tata', 'toyota', 'mahindra']
          else 68
      )
      st.markdown(
          f'🔥 **Market Demand / Hotness Meter:** **{hotness_score}%** (High'
          ' Resale Demand)'
      )
      st.progress(hotness_score / 100)

      st.markdown('---')
      st.subheader('📊 Quick Metrics')
      m1, m2 = st.columns(2)
      m1.metric('Model Accuracy ($R^2$)', f'{r2:.2f}')
      m2.metric('Prediction Error (MAE)', f'${mae:,.2f}')

    if predict_btn:

      def encode_val(col, val):
        if encoding_maps and col in encoding_maps:
          return encoding_maps[col].get(
              val, np.mean(list(encoding_maps[col].values()))
          )
        return 0

      feature_values = []
      for col in feature_names:
        if col == 'brand':
          feature_values.append(encode_val('brand', selected_brand))
        elif col == 'model':
          feature_values.append(encode_val('model', selected_model))
        elif col == 'city':
          feature_values.append(encode_val('city', selected_city))
        elif col == 'milage':
          feature_values.append(milage)
        elif col == 'car_age':
          feature_values.append(car_age)
        elif col == 'fuel_type':
          feature_values.append(encode_val('fuel_type', selected_fuel))
        elif col == 'transmission':
          feature_values.append(encode_val('transmission', selected_trans))
        elif col == 'accident':
          feature_values.append(encode_val('accident', selected_accident))
        else:
          feature_values.append(0)

      prediction = model.predict([feature_values])[0]
      estimated_new_price = prediction * 1.45

      health_score = 100 - (car_age * 4)
      if 'accident' in str(selected_accident).lower():
        health_score -= 15
      health_score = max(40, min(98, health_score))

      st.markdown('---')
      st.success(
          f'### 💰 Estimated Resale Price in {selected_city}:'
          f' **₹{prediction:,.0f}**'
      )

      col_p1, col_p2, col_p3 = st.columns(3)
      col_p1.metric('Used Car Valuation', f'₹{prediction:,.0f}')
      col_p2.metric('New Ex-Showroom Price (Est.)', f'₹{estimated_new_price:,.0f}')
      col_p3.metric(
          'Resale Health Score',
          f'{health_score} / 100',
          delta='Excellent' if health_score > 75 else 'Moderate',
      )

      report_text = f"""
=========================================
      CAR VALUATION REPORT & SUMMARY
=========================================
Brand: {selected_brand}
Model: {selected_model}
Location: {selected_city}
Fuel Type: {selected_fuel}
Transmission: {selected_trans}
Car Age: {car_age} Years
Mileage: {milage} km/l
Accident History: {selected_accident}
Market Demand Hotness: {hotness_score}%
-----------------------------------------
Estimated Resale Value: ₹{prediction:,.0f}
Estimated New Price: ₹{estimated_new_price:,.0f}
Resale Health Score: {health_score}/100
=========================================
Generated by Pro Car Price Prediction Dashboard
"""
      st.download_button(
          label='📥 Download Valuation Report (TXT)',
          data=report_text,
          file_name=f'{selected_brand}_{selected_model}_Valuation_Report.txt',
          mime='text/plain',
      )

  with tab2:
    st.subheader('⚖️ Comparative Model Analysis (Side-by-Side)')
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
      st.markdown('### Car Option A')
      c1_brand = st.selectbox('Brand A', brands, key='c1_b')
      c1_models = (
          sorted(df_raw[df_raw['brand'] == c1_brand]['model'].unique().tolist())
          if 'brand' in df_raw.columns
          else ['Model']
      )
      c1_model = st.selectbox('Model A', c1_models, key='c1_m')
      c1_age = st.slider('Age A (Years)', 0, 20, 2, key='c1_a')
      c1_mileage = st.number_input(
          'Mileage A (km/l)', 5.0, 40.0, 16.0, key='c1_mil'
      )

    with comp_col2:
      st.markdown('### Car Option B')
      c2_brand = st.selectbox('Brand B', brands, key='c2_b')
      c2_models = (
          sorted(df_raw[df_raw['brand'] == c2_brand]['model'].unique().tolist())
          if 'brand' in df_raw.columns
          else ['Model']
      )
      c2_model = st.selectbox('Model B', c2_models, key='c2_m')
      c2_age = st.slider('Age B (Years)', 0, 20, 4, key='c2_a')
      c2_mileage = st.number_input(
          'Mileage B (km/l)', 5.0, 40.0, 15.0, key='c2_mil'
      )


    def get_pred(b, m, age, mil):
      vals = []
      for col in feature_names:
        if col == 'brand':
          vals.append(
              encoding_maps.get('brand', {}).get(b, 0) if encoding_maps else 0
          )
        elif col == 'model':
          vals.append(
              encoding_maps.get('model', {}).get(m, 0) if encoding_maps else 0
          )
        elif col == 'city':
          vals.append(0)
        elif col == 'milage':
          vals.append(mil)
        elif col == 'car_age':
          vals.append(age)
        else:
          vals.append(0)
      return model.predict([vals])[0]


    if st.button('⚖️ Compare Both Cars'):
      res_a = get_pred(c1_brand, c1_model, c1_age, c1_mileage)
      res_b = get_pred(c2_brand, c2_model, c2_age, c2_mileage)

      col_res1, col_res2 = st.columns(2)
      col_res1.metric(
          f'{c1_brand} {c1_model} Valuation',
          f'₹{res_a:,.0f}',
          f'{c1_age} Yrs Old',
      )
      col_res2.metric(
          f'{c2_brand} {c2_model} Valuation',
          f'₹{res_b:,.0f}',
          f'{c2_age} Yrs Old',
      )

  with tab3:
    st.subheader('💰 EMI & Depreciation Financial Planner')
    col_e1, col_e2 = st.columns(2)
    with col_e1:
      st.markdown('### Monthly EMI Calculator')
      loan_amount = st.number_input(
          'Loan Amount (₹)',
          min_value=100000,
          max_value=5000000,
          value=800000,
          step=50000,
      )
      interest_rate = st.slider(
          'Interest Rate (% P.A)', min_value=5.0, max_value=20.0, value=9.5, step=0.5
      )
      tenure_years = st.slider(
          'Loan Tenure (Years)', min_value=1, max_value=7, value=5
      )

      r = interest_rate / (12 * 100)
      n = tenure_years * 12
      emi = (loan_amount * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1)
      st.success(f'**Estimated Monthly EMI:** ₹{emi:,.2f} / month')

    with col_e2:
      st.markdown('### 📉 5-Year Depreciation Trend')
      base_val = 1000000
      dep_years = [0, 1, 2, 3, 4, 5]
      dep_values = [base_val * (0.85 ** y) for y in dep_years]
      chart_data = pd.DataFrame(
          {'Year': dep_years, 'Estimated Value (₹)': dep_values}
      )
      st.line_chart(chart_data.set_index('Year'))

  with tab4:
    st.subheader('🔍 Feature Importance & Dashboard Insights')
    if feature_imp:
      imp_df = pd.DataFrame(
          list(feature_imp.items()), columns=['Feature', 'Importance']
      ).sort_values(by='Importance', ascending=True)
      st.bar_chart(imp_df.set_index('Feature'))

  with tab5:
    st.subheader('🤖 AI Car Expert Chatbot Assistant')
    st.markdown(
        'Ask any questions or type a car model name to get instant ML price'
        ' valuation!'
    )

    if 'messages' not in st.session_state:
      st.session_state.messages = [{
          'role': 'assistant',
          'content': (
              'Hello! I am your AI Car Expert Assistant. Type any car name (e.g.'
              ' "Mahindra Thar price" or "Honda Amaze") to get instant price'
              ' prediction!'
          ),
      }]

    for message in st.session_state.messages:
      with st.chat_message(message['role']):
        st.markdown(message['content'])

    if user_query := st.chat_input(
        'Type car name or question (e.g., Mahindra Thar price)...'
    ):
      st.session_state.messages.append({'role': 'user', 'content': user_query})
      with st.chat_message('user'):
        st.markdown(user_query)

      q_lower = user_query.lower()

      # Direct Machine Learning Prediction Integration inside Chatbot
      found_brand = next((b for b in brands if b.lower() in q_lower), None)
      models_list = (
          df_raw[df_raw['brand'] == found_brand]['model'].unique().tolist()
          if found_brand
          else df_raw['model'].unique().tolist()
      )
      found_model = next((m for m in models_list if m.lower() in q_lower), None)

      if found_brand or found_model or 'price' in q_lower or 'dam' in q_lower:
        b_use = found_brand if found_brand else brands[0]
        m_use = found_model if found_model else models_list[0]

        # Calculate ML Prediction for the requested car
        vals = []
        for col in feature_names:
          if col == 'brand':
            vals.append(
                encoding_maps.get('brand', {}).get(b_use, 0)
                if encoding_maps
                else 0
            )
          elif col == 'model':
            vals.append(
                encoding_maps.get('model', {}).get(m_use, 0)
                if encoding_maps
                else 0
            )
          elif col == 'city':
            vals.append(0)
          elif col == 'milage':
            vals.append(16.0)
          elif col == 'car_age':
            vals.append(3)
          else:
            vals.append(0)

        pred_val = model.predict([vals])[0]
        reply = (
            f'🤖 **AI Valuation Result:** For a standard **{b_use} {m_use}**'
            f' (approx. 3 years old), the estimated resale price is roughly'
            f' **₹{pred_val:,.0f}**!'
        )
      elif 'resale' in q_lower or 'sell' in q_lower:
        reply = (
            'To maximize your resale value, ensure timely service records,'
            ' keep mileage low, and maintain zero accident history.'
        )
      elif 'maintenance' in q_lower or 'service' in q_lower:
        reply = (
            'Regular engine oil replacement every 10,000 km and monthly tyre'
            ' inspections will keep your car in top shape.'
        )
      else:
        reply = (
            f"That's an interesting query! Try typing a specific brand or"
            ' model name like **"Mahindra Thar price"** or **"Honda Amaze"** to'
            ' get an instant price prediction.'
        )

      st.session_state.messages.append({'role': 'assistant', 'content': reply})
      with st.chat_message('assistant'):
        st.markdown(reply)