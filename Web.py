import streamlit as st
import numpy as np
import pickle
import datetime
import os


st.set_page_config(page_title="Copper", layout="wide", initial_sidebar_state="collapsed")
        
def app():
     
    with st.sidebar: 
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("Select options")
        choice = st.radio("Navigation", ["Home","Model"])
        st.info("This project application helps you predict the price and status")

    if choice == "Home":
        
            st.write('## **Problem Statement**')
            st.write('* The copper industry deals with less complex data related to sales and pricing.Where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer')
            st.write('* ML Regression model which predicts continuous variable Selling_Price.')
            st.write('* ML Classification model which predicts Status: Won or Loss')
            st.write('## Tools and Technologies used')
            st.write('Python, Streamlit, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Pickle, Streamlit-Option-Menu')
           
            st.write('## **USING MACHINE LEARNING**')

            st.write('#### REGRESSION - ExtraTreeRegressor')
            st.write('- Extra tree regressor is an ensemble supervised machine learning method that uses decision trees. It divide the data into subsets, that is, branches, nodes, and leaves. Like decision trees, regression trees select splits that decrease the dispersion of target attribute values. Thus, the target attribute values can be predicted from their mean values in the leaves.')
            st.write('#### CLASSIFICATION - RandomForestClassification')
            st.write('- Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption.')
            st.write('## **created by** \n Naveen Anandhan')
    if choice == "Model":
        
        task = st.selectbox('Select task', ['Regression', 'Classification'])
        
        if task == 'Regression':
            item_list = ['W', 'S', 'Others', 'PL', 'WI', 'IPL']
            status_list = ['Won', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
            country_list = ['28', '32', '38', '78', '27', '30', '25', '77', '39', '40', '26', '84', '80', '79', '113', '89']
            application_list = [10, 41, 28, 59, 15, 4, 38, 56, 42, 26, 27, 19, 20, 66, 29, 22, 40, 25, 67, 79, 3, 99, 2, 5, 39, 69, 70, 65, 58, 68]
            product_list = [1670798778, 1668701718, 628377, 640665, 611993, 1668701376, 164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374, 1282007633, 1668701698, 628117, 1690738206, 628112, 640400, 1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728, 1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819, 1665584320, 1665584662, 1665584642]

            st.subheader(":violet[Fill all the fields and press the button below to view the **Predicted price** of copper]")

            st.write('')
            st.write('')

            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                
                
                quantity = st.number_input('**Enter Quantity (Min:61 & Max:2349417) in tons**', min_value=61, max_value=2349417)
                thickness = st.number_input('**Enter Thickness (Min:0.18 & Max:400)**', min_value=0.18, max_value=400.0)
                width = st.number_input('**Enter Width (Min:1 & Max:2990)**', min_value=1, max_value=2990)

            with c2:
                country = st.selectbox('**Country Code**', country_list)
                status = st.selectbox('**Status**', status_list)
                item = st.selectbox('**Item Type**', item_list)

            with c3:
                application = st.selectbox('**Application Type**', application_list)
                product = st.selectbox('**Product Reference**', product_list)
                item_order_date = st.date_input("**Order Date**", datetime.date(2020, 7, 20))
                item_delivery_date = st.date_input("**Estimated Delivery Date**", datetime.date(2021, 12, 1))

            with c1:
                st.write('')
                st.write('')
                st.write('')
                
                # if st.button('PREDICT PRICE'):
                if st.button('PREDICT PRICE'):
                    try:
                        data = []

                        with open('et_reg.pkl', 'rb') as file:
                            et_reg = pickle.load(file)
                        with open('country.pkl', 'rb') as file:
                            encode_country = pickle.load(file)
                        with open('status.pkl', 'rb') as file:
                            encode_status = pickle.load(file)
                        with open('item_type.pkl', 'rb') as file:
                            encode_item = pickle.load(file)
                        with open('scaling.pkl', 'rb') as file:
                            scaled_data_reg = pickle.load(file)

                        country_to_transformed = dict(zip(country_list, encode_country))
                        item_to_transformed = dict(zip(item_list, encode_item))
                        status_to_transformed = dict(zip(status_list, encode_status))

                        encoded_ct = country_to_transformed.get(country)
                        if encoded_ct is None:
                            st.error("Country not found.")
                            st.stop()

                        encode_it = item_to_transformed.get(item)
                        if encode_it is None:
                            st.error("Item type not found.")
                            st.stop()

                        encode_st = status_to_transformed.get(status)
                        if encode_st is None:
                            st.error("Status not found.")
                            st.stop()

                        order = datetime.datetime.strptime(str(item_order_date), "%Y-%m-%d")
                        delivery = datetime.datetime.strptime(str(item_delivery_date), "%Y-%m-%d")
                        day = (delivery - order).days

                        data.extend([quantity, thickness, width, encoded_ct, encode_st, encode_it, application, product, day])

                        x = np.array(data).reshape(1, -1)
                        new_sample_scaled = scaled_data_reg.transform(x)
                        new_pred = et_reg.predict(new_sample_scaled)
                        predicted_price = new_pred[0]

                        st.success(f'**Predicted Selling Price: :green[₹] {predicted_price:.2f}**')
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        
        if task == 'Classification':
 
            item_list_cls = ['W', 'S', 'Others', 'PL', 'WI', 'IPL']
            country_list_cls = ['28', '32', '38', '78', '27', '30', '25', '77', '39', '40', '26', '84', '80', '79', '113', '89']
            application_list_cls = [10, 41, 28, 59, 15, 4, 38, 56, 42, 26, 27, 19, 20, 66,
                                    29, 22, 40, 25, 67, 79, 3, 99, 2, 5, 39, 69, 70, 65, 58, 68]
            product_list_cls = [1670798778, 1668701718, 628377, 640665, 611993, 1668701376,
                                    164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374,
                                    1282007633, 1668701698, 628117, 1690738206, 628112, 640400,
                                    1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728,
                                    1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819,
                                    1665584320, 1665584662, 1665584642]
            

            st.subheader(":violet[Fill all the fields and press the button below to view the **Predicted status** (WON/LOST) of copper]")

            cc1, cc2, cc3 = st.columns([2, 2, 2])
            with cc1:
                quantity_cls = st.number_input('**Enter Quantity (Min:61 & Max:2349417) in tons**', min_value=61, max_value=2349417)
                thickness_cls = st.number_input('**Enter Thickness (Min:0.18 & Max:400)**', min_value=0.18, max_value=400.0)
                width_cls = st.number_input('**Enter Width (Min:1 & Max:2990)**', min_value=1, max_value=2990)

            with cc2:
                selling_price_cls = st.number_input('**Enter Selling Price (Min:1, Max:100001015)**', min_value=1, max_value=100001015)
                item_cls = st.selectbox('**Item Type**', item_list_cls)
                country_cls = st.selectbox('**Country Code**', country_list_cls)

            with cc3:
                application_cls = st.selectbox('**Application Type**', application_list_cls)
                product_cls = st.selectbox('**Product Reference**', product_list_cls)
                item_order_date_cls = st.date_input("**Order Date**", datetime.date(2020, 7, 20))
                item_delivery_date_cls = st.date_input("**Estimated Delivery Date**", datetime.date(2021, 12, 1))

            with cc1:
                st.write('')
                st.write('')
                st.write('')
                
                if st.button('PREDICT STATUS'):
                    try:
                        data_cls = []

                        with open('country.pkl', 'rb') as file:
                            encode_country_cls = pickle.load(file)
                        with open('status.pkl', 'rb') as file:
                            encode_status_cls = pickle.load(file)
                        with open('item_type.pkl', 'rb') as file:
                            encode_item_cls = pickle.load(file)
                        with open('scaling_classify.pkl', 'rb') as file:
                            scaled_data_cls = pickle.load(file)
                        with open('RF_class.pkl', 'rb') as file:
                            trained_model_cls = pickle.load(file)

                        country_to_transformed_cls = dict(zip(country_list_cls, encode_country_cls))
                        item_to_transformed_cls = dict(zip(item_list_cls, encode_item_cls))

                        encoded_ct_cls = country_to_transformed_cls.get(country_cls)
                        if encoded_ct_cls is None:
                            st.error("Country not found.")
                            st.stop()

                        encode_it_cls = item_to_transformed_cls.get(item_cls)
                        if encode_it_cls is None:
                            st.error("Item type not found.")
                            st.stop()

                        order_cls = datetime.datetime.strptime(str(item_order_date_cls), "%Y-%m-%d")
                        delivery_cls = datetime.datetime.strptime(str(item_delivery_date_cls), "%Y-%m-%d")
                        day_cls = (delivery_cls - order_cls).days

                        data_cls.extend([quantity_cls, thickness_cls, width_cls, selling_price_cls, encoded_ct_cls, encode_it_cls, application_cls, product_cls, day_cls])

                        x_cls = np.array(data_cls).reshape(1, -1)
                        new_sample_scaled_cls = scaled_data_cls.transform(x_cls)
                        new_pred_cls = trained_model_cls.predict(new_sample_scaled_cls)
                        predicted_status = new_pred_cls[0]

                        if predicted_status == 7:
                            st.success(f'**Predicted Status: :green[WON]**')
                        else:
                            st.error(f'**Predicted Status: :red[LOST]**')

                    except Exception as e:
                        st.error(f"An error occurred: {e}")


                

        
if __name__ == '__main__':
    app()
