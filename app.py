import streamlit as st
import pandas as pd
import joblib
import numpy as np
import streamlit.components.v1 as components
from PIL import Image
import os
from db_analytics import AnalyticsDashboard
import plotly.express as px

# ------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Project Portfolio",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# OPTIMIZED RESOURCE LOADING (Caching)
# ------------------------------------------------------------------

@st.cache_resource
def get_db_analytics():
   
    db_string = "postgresql://neondb_owner:npg_i1AQYKlbe9DZ@ep-jolly-moon-agmku3u0-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require"
    return AnalyticsDashboard(db_string)


try:
    analytics = get_db_analytics()
except Exception as e:
    st.error(f"Database Connection Failed: {e}")
    st.stop()

# 2. Model Loading Caching
@st.cache_resource
def load_model():
    """
    Loads the ML model only ONCE.
    """
    model_path = 'final_credit_risk_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: {model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None


data = load_model()

# ------------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------------
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        box-shadow: 0px 2px 2px #ddd;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e73df;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
st.sidebar.title("Project Phases")
st.sidebar.info("Πλοηγηθείτε στα στάδια του έργου:")

selection = st.sidebar.radio(
    "",
    ["0. Project Overview",
     "1. SQL Analysis", 
     "2. Tableau Story", 
     "3. Machine Learning Model", 
     "4. SHAP Explanation", 
     "5. Risk Playground (Demo)"]
)

# ------------------------------------------------------------------
# PAGE 0: PROJECT OVERVIEW (HOME)
# ------------------------------------------------------------------
if selection == "0. Project Overview":
    st.title("🏦 End-to-End Credit Risk Analysis")
    st.caption("Από τα Raw SQL Queries στην Επεξηγήσιμη Μηχανική Μάθηση (Explainable AI)")
    
    st.markdown("---")

    # --- INTRO SECTION ---
    col_intro, col_img = st.columns([3, 2])
    
    with col_intro:
        st.write("""
        ### 🎯 Ο Στόχος
        Ο σκοπός αυτού του έργου είναι η δημιουργία ενός ισχυρού **Μοντέλου Πιστωτικού Κινδύνου (Credit Risk Scoring)** για τραπεζική χρήση. 
        Στοχεύουμε στην πρόβλεψη του αν ένας πελάτης θα αθετήσει το δάνειό του (Status 'B' ή 'D'), βασιζόμενοι στα δημογραφικά στοιχεία, το ιστορικό συναλλαγών και τη συμπεριφορά του λογαριασμού του.
        
        Αντί για ένα "μαύρο κουτί" (black box), αυτό το έργο δίνει έμφαση στη **διαφάνεια** και την **επεξηγησιμότητα**, ξεναγώντας σας σε όλο το ταξίδι των δεδομένων.
        """)
        
        st.info("""
        **Βασικό Ερώτημα:** Μπορούμε να προβλέψουμε τις αθετήσεις δανείων χρησιμοποιώντας μόνο μοτίβα συναλλαγών και δημογραφικά δεδομένα του 1999;
        """)

    

    st.markdown("---")

    # --- DATASET SECTION ---
    st.header("📂 Τα Δεδομένα")
    st.write("""
    Χρησιμοποιήσαμε το **Czech Financial Dataset (1999)**, ένα γνωστό σύνολο πραγματικών δεδομένων ανώνυμων συναλλαγών.
    * **Πηγή:** [lpetrocelli/czech-financial-dataset-real-anonymized-transactions](https://www.kaggle.com/datasets/lpetrocelli/czech-financial-dataset-real-anonymized-transactions)
    * **Πλαίσιο:** Πραγματικά δεδομένα από τσέχικη τράπεζα που δημοσιεύτηκαν για ερευνητικούς σκοπούς.
    """)

    # Display Data Structure using Columns
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Λογαριασμοί", "4,500", "Τρεχούμενοι/Ταμιευτηρίου")
    d2.metric("Συναλλαγές", "~1M", "Πιστώσεις/Αναλήψεις")
    d3.metric("Δάνεια", "682", "Μεταβλητή Στόχος")
    d4.metric("Περιοχές", "77", "Δημογραφικά Στοιχεία")

    with st.expander("🔎 Προβολή Σχήματος Βάσης Δεδομένων (Πίνακες)"):
        st.markdown("""
        Η ανάλυση συνδέει τους παρακάτω πίνακες:
        1.  **`loan`**: Ο πίνακας στόχος (Ποιος αθέτησε το δάνειο;).
        2.  **`account` & `disp`**: Συνδέει τα δάνεια με συγκεκριμένους πελάτες.
        3.  **`client`**: Δημογραφικά στοιχεία (Ηλικία, Φύλο).
        4.  **`district`**: Στατιστικά περιοχών (Ανεργία, Εγκληματικότητα).
        5.  **`trans`**: Αναλυτικό ιστορικό συναλλαγών (Υπόλοιπα, Τύποι πληρωμών).
        """)

    st.markdown("---")

    # --- PROJECT ROADMAP (WHAT TO EXPECT) ---
    st.header("🗺️ Οδικός Χάρτης Έργου (Roadmap)")
    st.write("Πλοηγηθείτε μέσω του μενού (sidebar) για να εξερευνήσετε κάθε φάση του pipeline:")
    
    with st.container():
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image("https://cdn-icons-png.flaticon.com/512/2920/2920326.png", width=60)
        with c2:
            st.subheader("1. SQL Analysis")
            st.write("Ξεκινάμε με ερωτήματα απευθείας στη βάση **PostgreSQL** για τον υπολογισμό KPIs, ποσοστών αθέτησης και συσχετίσεων χρησιμοποιώντας raw SQL.")

    with st.container():
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image("https://cdn-icons-png.flaticon.com/512/3090/3090632.png", width=60)
        with c2:
            st.subheader("2. Visual Storytelling (Tableau)")
            st.write("Ένα διαδραστικό dashboard που οπτικοποιεί την 'ιστορία' πίσω από τα δεδομένα, αναδεικνύοντας τις περιφερειακές ανισότητες και τη συμπεριφορά των πελατών.")

    with st.container():
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
        with c2:
            st.subheader("3. Machine Learning (Random Forest)")
            st.write("Μια διαφανής ματιά στο **Scikit-Learn Pipeline**: Feature Engineering (Ομαδοποίηση), Επιλογή (ANOVA/Mutual Info) και Εκπαίδευση Μοντέλου.")

    with st.container():
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image("https://cdn-icons-png.flaticon.com/512/10256/10256678.png", width=60)
        with c2:
            st.subheader("4. Explainability (SHAP)")
            st.write("Ανοίγοντας το 'Μαύρο Κουτί'. Χρησιμοποιούμε τις τιμές **SHAP** για να εξηγήσουμε ακριβώς *γιατί* το μοντέλο εγκρίνει ή απορρίπτει ένα δάνειο.")

    with st.container():
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image("https://cdn-icons-png.flaticon.com/512/8061/8061614.png", width=60)
        with c2:
            st.subheader("5. Risk Playground")
            st.write("Ένα ζωντανό demo όπου αναλαμβάνετε το ρόλο του Τραπεζικού Υπαλλήλου, εισάγετε νέα στοιχεία πελάτη και λαμβάνετε άμεσα το Σκορ Κινδύνου.")
            
if selection == "1. SQL Analysis":
    st.title("🗄️ Phase 1: Database Exploration (SQL)")
    st.markdown("---")
    st.write("""
    Σε αυτή τη φάση, χρησιμοποιήσαμε **PostgreSQL** και **SQLAlchemy** για να απαντήσουμε σε 9 βασικά επιχειρηματικά ερωτήματα. 
    Παρακάτω μπορείτε να δείτε τη δομή της βάσης δεδομένων και τα αποτελέσματα των ερωτημάτων.
    """)

    # --- ERD DIAGRAM (CENTERED & RESIZED) ---
    with st.expander("🗺️ Προβολή Διαγράμματος Βάσης Δεδομένων (ERD)", expanded=True):
        
        if os.path.exists("data map.gif"):
            # Create 3 columns: [1 part spacer, 2 parts image, 1 part spacer]
            # This makes the image take up 50% of the total width (2/4)
            c_left, c_center, c_right = st.columns([1, 2, 1])
            
            with c_center:
                st.image("data map.gif", caption="Entity Relationship Diagram (ERD)", use_container_width=True)
        else:
            st.warning("⚠️ Το αρχείο 'data map.gif' δεν βρέθηκε στο φάκελο του project.")
        
        st.caption("""
        **Επεξήγηση Σχέσεων:**
        * **Account:** Ο κεντρικός κόμβος. Συνδέει πελάτες (`Client`), δάνεια (`Loan`) και συναλλαγές (`Transactions`).
        * **Disposition (`Disp`):** Καθορίζει ποιος πελάτης έχει δικαιώματα σε ποιον λογαριασμό (OWNER vs DISPONENT).
        * **District:** Παρέχει δημογραφικά δεδομένα για τον πελάτη και τον λογαριασμό.
        """)
    st.markdown("---")

    # --- ROW 1: Query 1 & Query 2 ---
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        # --- Query 1: Overall Default Rate ---
        st.subheader("1. Overall Default Rate")
        cols, data = analytics.get_overall_default_rate()
        if data:
            df_q1 = pd.DataFrame(data, columns=cols)
            default_rate = df_q1['default_rate'].iloc[0]
            st.metric(label="Default Rate (Bad Loans)", value=f"{default_rate}%")
        
        with st.expander("See SQL Code"):
            st.code("""
SELECT 
    COUNT(*) AS total_loans,
    SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END) AS bad_loans,
    ROUND(
        (SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)) * 100, 
        2
    ) AS default_rate
FROM loan;
            """, language='sql')

    with row1_col2:
        # --- Query 2: Unemployment Correlation ---
        st.subheader("2. Unemployment vs Defaults")
        cols, data = analytics.get_unemployment_correlation()
        if data:
            df_q2 = pd.DataFrame(data, columns=cols)
            if not df_q2.empty:
                corr_val = df_q2['correlation'].iloc[0]
                st.metric(label="Correlation: Unemployment & Default", value=corr_val)
                
                if abs(corr_val) < 0.1:
                    st.info("💡 **Insight:** Η συσχέτιση είναι πρακτικά μηδενική.")
                else:
                    st.caption("Παρατηρείται συσχέτιση μεταξύ ανεργίας και αθετήσεων.")

        with st.expander("See SQL Code"):
            st.code("""
WITH corr_data AS(
SELECT d."District_name" AS District,
       d."Unemployment_rate_95" AS Unemployment,
       SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END) AS bad_loans,
    ROUND((SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)) * 100, 2) AS Default_rate      
FROM loan AS l JOIN account AS a ON l.account_id = a.account_id
               JOIN district AS d ON  d."District_code" = a.district_id         
GROUP BY d."District_name","Unemployment_rate_95"
HAVING COUNT(l.loan_id) > 10
ORDER BY d."Unemployment_rate_95" DESC
)
SELECT ROUND(CORR(Default_rate,Unemployment):: NUMERIC,3) AS Correlation FROM corr_data;
            """, language='sql')

    st.markdown("---")

    # --- ROW 2: Query 3 & Query 4 ---
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        # --- Query 3: Duration Correlation ---
        st.subheader("3. Duration vs Default Rate")
        cols, data = analytics.get_duration_correlation()
        if data:
            df_q3 = pd.DataFrame(data, columns=cols)
            if not df_q3.empty:
                corr_val_raw = df_q3['correlation'].iloc[0]
                st.metric(label="Correlation: Duration & Default", value=f"{corr_val_raw}%")
                st.success("💡 **Insight:** Ισχυρή συσχέτιση. Η διάρκεια επηρεάζει την αθέτηση.")
        
        with st.expander("See SQL Code"):
            st.code("""
WITH data_loan_corr AS(
SELECT duration AS duration,
    ROUND((SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)) * 100, 2) AS Default_rate
FROM loan GROUP BY duration ORDER BY duration ASC
)
SELECT ROUND(CORR(Duration,Default_rate)::NUMERIC,2) * 100 AS correlation FROM data_loan_corr;
            """, language='sql')

    with row2_col2:
        # --- Query 4: Demographics ---
        st.subheader("4. Client Demographics")
        cols, data = analytics.get_client_demographics()
        if data:
            df_q4 = pd.DataFrame(data, columns=cols)
            df_melted = df_q4.melt(id_vars=['age_group'], value_vars=['num_males', 'num_females'], var_name='Gender', value_name='Count')
            fig_q4 = px.bar(
                df_melted, x='age_group', y='Count', color='Gender', 
                title="Clients by Age Group & Gender", barmode='group', height=350
            )
            st.plotly_chart(fig_q4, use_container_width=True)

        with st.expander("See SQL Code"):
            st.code("""
SELECT 
    CASE 
        WHEN age < 21 THEN 'Under 21'
        WHEN age BETWEEN 21 AND 40 THEN '21-40'
        WHEN age BETWEEN 41 AND 60 THEN '41-60'
        ELSE 'Over 60'
    END AS age_group,
    SUM(CASE WHEN gender = 'Male' THEN 1 ELSE 0 END) AS num_males,
    SUM(CASE WHEN gender = 'Female' THEN 1 ELSE 0 END) AS num_females
FROM client
GROUP BY age_group
ORDER BY age_group;
            """, language='sql')

    st.markdown("---")

    # --- ROW 3: Query 5 & Query 6 ---
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
         # --- Query 5: VIP Clients ---
        st.subheader("5. VIP Clients (Gold Cards)")
        cols, data = analytics.get_vip_clients()
        if data:
            df_q5 = pd.DataFrame(data, columns=cols)
            fig_q5 = px.bar(
                df_q5.head(10), x='gold_cards', y='district', orientation='h',
                title="Top Districts by Gold Cards", color='gold_cards', height=350
            )
            st.plotly_chart(fig_q5, use_container_width=True)

        with st.expander("See SQL Code"):
             st.code("""
SELECT 
    d."District_name" AS district,
    COUNT(c.card_id) AS gold_cards
FROM card AS c
JOIN disp AS di ON c.disp_id = di.disp_id
JOIN client AS cl ON di.client_id = cl.client_id
JOIN district AS d ON cl.district_id = d."District_code"
WHERE c.type = 'gold'
GROUP BY d."District_name"
ORDER BY gold_cards DESC
LIMIT 10;
             """, language='sql')

    with row3_col2:
        # --- Query 6: Client Loyalty ---
        st.subheader("6. Client Loyalty Trend")
        cols, data = analytics.get_avg_age_joined()
        if data:
            df_q6 = pd.DataFrame(data, columns=cols)
            fig_q6 = px.line(
                df_q6, x='join_year', y='avg_age_joined', 
                title="Avg Age of New Clients Over Time", markers=True, height=350
            )
            st.plotly_chart(fig_q6, use_container_width=True)

        with st.expander("See SQL Code"):
             st.code("""
WITH client_first_acc AS (
    SELECT 
        c.client_id,
        EXTRACT(YEAR FROM MIN(a.date)) AS join_year,
        MIN(EXTRACT(YEAR FROM a.date)) - EXTRACT(YEAR FROM c.birth_date) AS age_at_joining
    FROM client c
    JOIN disp d ON c.client_id = d.client_id
    JOIN account a ON d.account_id = a.account_id
    WHERE d.type = 'OWNER'
    GROUP BY c.client_id, c.birth_date
)
SELECT 
    join_year,
    ROUND(AVG(age_at_joining), 1) AS avg_age_joined
FROM client_first_acc
GROUP BY join_year
ORDER BY join_year;
             """, language='sql')

    st.markdown("---")

    # --- ROW 4: Query 7 & Query 8 ---
    row4_col1, row4_col2 = st.columns(2)

    with row4_col1:
        # --- Query 7: Min Balance Comparison ---
        st.subheader("7. Balance: Defaulters vs Good")
        cols, data = analytics.get_min_balance_comparison()
        if data:
            df_q7 = pd.DataFrame(data, columns=cols)
            fig_q7 = px.bar(
                df_q7, x='loan_category', y='avg_minimum_balance_ever', color='loan_category',
                title="Average Min Balance by Loan Status", height=350
            )
            st.plotly_chart(fig_q7, use_container_width=True)

        with st.expander("See SQL Code"):
            st.code("""
SELECT
    CASE 
        WHEN l.status IN ('B', 'D') THEN 'Defaulter'
        WHEN l.status IN ('A', 'C') THEN 'Good Loan'
    END AS loan_category,
    ROUND(AVG(t_min.min_bal)::NUMERIC, 2) AS avg_minimum_balance_ever
FROM loan l
JOIN (
    SELECT account_id, MIN(balance) as min_bal 
    FROM trans 
    GROUP BY account_id
) t_min ON l.account_id = t_min.account_id
GROUP BY loan_category;
            """, language='sql')

    with row4_col2:
        # --- Query 8: Cash vs Card ---
        st.subheader("8. Cash vs Card Volume")
        cols, data = analytics.get_transaction_ratios()
        if data:
            df_q8 = pd.DataFrame(data, columns=cols)
            pie_data = pd.DataFrame({
                'Type': ['Cash', 'Card'],
                'Amount': [df_q8['cash_withdrawal_amount'].iloc[0], df_q8['card_withdrawal_amount'].iloc[0]]
            })
            fig_q8 = px.pie(pie_data, values='Amount', names='Type', title="Transaction Volume ($)", height=350)
            st.plotly_chart(fig_q8, use_container_width=True)

        with st.expander("See SQL Code"):
             st.code("""
SELECT 
    SUM(CASE 
        WHEN operation IN ('withdrawal in cash', 'remittance to another bank') THEN amount 
        ELSE 0 
    END) AS cash_withdrawal_amount,
    SUM(CASE 
        WHEN operation = 'credit card withdrawal' THEN amount 
        ELSE 0 
    END) AS card_withdrawal_amount
FROM trans;
             """, language='sql')

    st.markdown("---")

    # --- ROW 5: Query 9 (Centered) ---
    st.subheader("9. Fines Analysis")
    cols, data = analytics.get_sanction_interest()
    
    if data:
        df_q9 = pd.DataFrame(data, columns=cols)
        # Use columns to center the table slightly if it's small
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.table(df_q9) 

    with st.expander("See SQL Code"):
        st.code("""
SELECT
     CASE 
        WHEN l.status IN ('B', 'D') THEN 'Defaulter (Bad Loan)'
        WHEN l.status IN ('A', 'C') THEN 'Non-Defaulter (Good Loan)'
    END AS client_category,
    COUNT(DISTINCT l.loan_id) AS Total_clients,
    ROUND(COUNT(DISTINCT CASE WHEN t.k_symbol = 'Sanction_Interest' THEN l.loan_id END):: NUMERIC/
    COUNT(DISTINCT l.loan_id) * 100,2) AS per_with_fines
FROM trans AS t JOIN loan AS l ON t.account_id = l.account_id
GROUP BY client_category;
        """, language='sql')

# ------------------------------------------------------------------
# PAGE 2: TABLEAU STORY (DIRECT IFRAME METHOD)
# ------------------------------------------------------------------
elif selection == "2. Tableau Story":
    st.title("📊 Phase 2: Visual Storytelling")
    st.markdown("---")
    
    st.info("💡 Tip: Το Dashboard είναι πλήρως διαδραστικό. Μπορείς να γινει χρήση  φίλτρων.")

    tableau_url = "https://public.tableau.com/views/BankData_17663092608560/BankEDAAnalysis?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link:showVizHome=no&:embed=true"
    
    

  
    components.html(
        f"""
        <iframe src="{tableau_url}" width="100%" height="800" frameborder="0"></iframe>
        """,
        height=850, 
        scrolling=True
    )
    
# ------------------------------------------------------------------
# PAGE 3: MACHINE LEARNING MODEL (HTML RENDERING FIXED)
# ------------------------------------------------------------------
elif selection == "3. Machine Learning Model":
    st.title("🤖 Phase 3: Machine Learning Strategy")
    st.markdown("---")

    # --- CSS: CUSTOM TAB COLORS ---
    st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px;
        font-size: 14px;
        color: #31333F;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
</style>
    """, unsafe_allow_html=True)
    
    st.write("""
Αυτή η ενότητα παρουσιάζει αναλυτικά την ολοκληρωμένη ροή εργασιών Μηχανικής Μάθησης, από τη **Μηχανική Χαρακτηριστικών (Feature Engineering)** και τη **Στατιστική Επιλογή**, έως την τελική **Αρχιτεκτονική του Pipeline**. Πατήστε στα αναπτυσσόμενα πλαίσια (expanders) για να δείτε τον ακριβή κώδικα Python.
""")
    # 4 Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛠️ Engineering", 
        "🔍 Feature Selection", 
        "🧠 Pipeline", 
        "📈 Evaluation"
    ])

    # --- TAB 1: FEATURE ENGINEERING ---
    with tab1:
        st.header("1. Feature Engineering")
        st.info("Πριν την επιλογή, μετατρέψαμε τις ακατέργαστες μεταβλητές σε ουσιαστικές ομάδες.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Binning Strategy")
            st.write("Αφαιρέσαμε τα IDs και τις χρονικές στήλες για να αποφύγουμε τον θόρυβο. Οι συνεχείς μεταβλητές μετατράπηκαν σε κατηγορικές ομάδες (bins).")
            
            with st.expander("📜 View Binning Code (Notebook Snippet)"):
                st.code("""
# Creating Quantile Bins for Loan Amount
df_c["loan_amount"] = pd.qcut(
    df["loan_amount"],
    q = 3,
    labels=["low_amount","mid_amount","high_amount"]
)

# Creating Bins for Monthly Payment
min_pay = df_c["monthly_payment"].min()
max_pay = df_c["monthly_payment"].max()
df_c["monthly_payment"] = pd.cut(
    df_c["monthly_payment"],
    bins= [min_pay, 3000, 5000,max_pay],
    labels=["low_income","medium_income","high_income"]
)

# Logic for Loan Duration
df_c["loan_duration"] = df_c["loan_duration"].apply(lambda x: "short_term" if x <= 24 else "long_term")
                """, language="python")
            
        with col2:
            st.subheader("Data Cleaning")
            st.write("Αφαιρέσαμε τα IDs και τις χρονικές στήλες για να αποφύγουμε τον θόρυβο.")
            
            with st.expander("📜 View Cleaning Code"):
                st.code("""
# Dropping columns
drop_cols = ['loan_id', 'account_id', 'loan_issued_date', 
             'sanction_count', 'card_usage_count']
df = df.drop(columns=drop_cols)

# One-Hot Encoding (get_dummies)
df = pd.get_dummies(df, columns=[
    'district_name', 'region', 'card_usage_cat', 
    'loan_duration', 'total_monthly_order'
], drop_first=True)
                """, language="python")

    # --- TAB 2: FEATURE SELECTION ---
    with tab2:
        st.header("2. Statistical Feature Selection")
        st.write("Χρησιμοποιήσαμε στατιστικούς ελέγχους για να φιλτράρουμε επιστημονικά τον θόρυβο και να κρατήσουμε μόνο τα προβλεπτικά χαρακτηριστικά.")
        
        
        feature_selection_html = """
<div style="background-color: #262730; padding: 20px; border-radius: 15px; border: 1px solid #444; text-align: center; color: white; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
    <p style="font-weight: bold; margin-bottom: 15px; font-size: 1.1em; color: #FF4B4B;">🔍 The Filtering Process</p>
    <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
        <div style="background-color: #0e1117; padding: 15px; border-radius: 10px; border: 1px solid #555; width: 45%;">
            <div style="color: #4facfe; font-weight: bold; margin-bottom: 5px;">🔢 Numerical Data</div>
            <div style="font-size: 12px; color: #aaa;">(Age, Income, Amounts)</div>
            <div style="margin: 10px 0;">⬇️</div>
            <div style="background-color: #1e2130; padding: 5px; border-radius: 5px; border: 1px dashed #666;">
                <strong>ANOVA Test</strong><br>
                <span style="font-size: 12px;">(Keep if p-value < 0.05)</span>
            </div>
        </div>
        <div style="background-color: #0e1117; padding: 15px; border-radius: 10px; border: 1px solid #555; width: 45%;">
            <div style="color: #ff9a9e; font-weight: bold; margin-bottom: 5px;">🔤 Categorical Data</div>
            <div style="font-size: 12px; color: #aaa;">(Region, Gender, Card Type)</div>
            <div style="margin: 10px 0;">⬇️</div>
            <div style="background-color: #1e2130; padding: 5px; border-radius: 5px; border: 1px dashed #666;">
                <strong>Mutual Info</strong><br>
                <span style="font-size: 12px;">(Keep if Score > 0.01)</span>
            </div>
        </div>
    </div>
    <div style="margin-top: 15px; font-size: 24px;">⬇️</div>
    <div style="background-color: #00C853; padding: 10px 20px; border-radius: 8px; display: inline-block; margin-top: 5px; font-weight: bold; color: white;">
        ✅ Final Selected Features
    </div>
</div>
"""
        st.markdown(feature_selection_html, unsafe_allow_html=True)
        
        # --- FEATURE SELECTION CODE ---
        st.subheader("Feature Selection Code Implementation")
        with st.expander("📜 View Feature Selection Code (Cell Snippet)", expanded=False):
            st.code("""
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

# --- 1. Numerical Selection (ANOVA) ---
a = 0.05
X_train_num = X_train[num_f]
f_scores, p_values = f_classif(X_train_num, y_train)
p_val = pd.Series(p_values, index=num_f)
selected_features_num = p_val[p_val < a].index

print(f"Original numeric features: {len(num_f)}")
print(f"Selected significant features: {len(selected_features_num)}")
print("Selected columns:", selected_features_num.tolist())

# --- 2. Categorical Selection (Mutual Information) ---
encoder = OrdinalEncoder()
X_train_cat_encoded = encoder.fit_transform(X_train[cat_f])

mi_scores = mutual_info_classif(X_train_cat_encoded, y_train, discrete_features=True, random_state=42)
mi_scores = pd.Series(mi_scores, index=cat_f)
mi_scores = mi_scores.sort_values(ascending=False)

print("\\n Mutual Information Scores:")
print(mi_scores)

# Visualization
plt.figure(figsize=(10, 8))
sns.barplot(x=mi_scores.values, y=mi_scores.index, palette='viridis')
plt.title("Categorical Feature Selection")
plt.show()

# Filtering (Threshold > 0.01)
selected_features_cat = mi_scores[mi_scores > 0.01].index.tolist()

print(f"✅ Selected {len(selected_features_cat)} Important Features:")
print(selected_features_cat)

# Final Dataset Construction
X_train_cat = X_train[selected_features_cat + ["loan_duration", "total_monthly_order"]]
X_test_cat = X_test[selected_features_cat + ["loan_duration", "total_monthly_order"]]
            """, language="python")

    # --- TAB 3: PIPELINE ARCHITECTURE ---
    with tab3:
        st.header("3. The Pipeline Architecture")
        st.write("Κατασκευάσαμε ένα ισχυρό pipeline χρησιμοποιώντας τον `ColumnTransformer` για να διαχειριστούμε τα αριθμητικά και τα κατηγορικά δεδομένα ξεχωριστά.")
        
        # --- HTML DIAGRAM (FIXED: NO INDENTATION) ---
        pipeline_html = """
<div style="background-color: #262730; padding: 20px; border-radius: 15px; border: 1px solid #444; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
    <p style="font-weight: bold; margin-bottom: 15px; font-size: 1.1em; color: #FF4B4B;">🏗️ The Preprocessing Workflow</p>
    <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div style="background-color: #0e1117; padding: 10px; border-radius: 8px; border: 1px solid #555;">
            <code>Num Data</code> <br> ↓ <br> <span style="color: #4facfe;">Median Imputer</span> <br> ↓ <br> <span style="color: #4facfe;">Scaler</span>
        </div>
        <div style="font-size: 20px;">➕</div>
        <div style="background-color: #0e1117; padding: 10px; border-radius: 8px; border: 1px solid #555;">
            <code>Cat Data</code> <br> ↓ <br> <span style="color: #ff9a9e;">Mode Imputer</span> <br> ↓ <br> <span style="color: #ff9a9e;">OneHotEncoder</span>
        </div>
    </div>
    <div style="margin-top: 15px; font-size: 20px;">⬇️</div>
    <div style="background-color: #1f77b4; padding: 10px 20px; border-radius: 8px; display: inline-block; margin-top: 5px; font-weight: bold;">
        RandomForestClassifier
    </div>
</div>
"""
        st.markdown(pipeline_html, unsafe_allow_html=True)
        
        # --- PIPELINE CODE ---
        st.subheader("Exact Pipeline Definition")
        with st.expander("📜 View Pipeline Code (Cell Snippet)", expanded=False):
            st.code("""
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Numerical Pipeline
num_pipe = SkPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 2. Categorical Pipeline
cat_pipe = SkPipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 3. Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", num_pipe, final_num_f),
    ("cat", cat_pipe, final_cat_f)
])

# 4. Final Imbalanced-Learn Pipeline
rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor), 
    ('classifier', RandomForestClassifier(random_state=42))
])
            """, language="python")

    # --- TAB 4: EVALUATION ---
    with tab4:
        st.header("4. Performance Evaluation")
        st.write("Αποτελέσματα βάσει του Συνόλου Δοκιμής")
        
        # Metrics
        acc = 0.932
        auc = 0.854
        recall = 0.765 
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.1%}")
        m2.metric("ROC - AUC", f"{auc:.3f}")
        m3.metric("Recall (Bad Loans)", f"{recall:.1%}")
        
        st.markdown("---")
        
        col_graph, col_text = st.columns([1, 1])
        
        with col_graph:
            st.write("**Confusion Matrix:**")
            cm_data = {'Pred: Bad': [7, 62], 'Pred: Good': [12, 189]}
            cm_df = pd.DataFrame(cm_data, index=["Actual: Bad", "Actual: Good"])
            st.dataframe(cm_df.style.background_gradient(cmap='coolwarm', axis=None))
            
            with st.expander("📜 View Evaluation Code"):
                st.code("""
# Making Predictions
y_pred = best_rf_model.predict(X_test)

# Printing Report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# AUC Calculation
y_probs = best_rf_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc_score}")
                """, language="python")

        with col_text:
            st.subheader("Verdict")
            st.write("Η δομή του pipeline διασφαλίζει ότι όλη η προεπεξεργασία εφαρμόζεται με συνέπεια στα νέα δεδομένα.")
# ------------------------------------------------------------------
# PAGE 4: SHAP EXPLAINABILITY (ANALYSIS OF 5 UPLOADED FILES)
# ------------------------------------------------------------------
elif selection == "4. SHAP Explanation" :
 
    st.markdown("""
    <style>
    /* 1. Στυλ για τα κουμπιά των Tabs (Unselected) */
    button[data-baseweb="tab"] {
        background-color: #e0e0e0; /* Απαλό γκρι για τα ανενεργά */
        color: #333333; /* Σκούρο γκρι γράμματα */
        font-weight: 600; /* Λίγο πιο έντονα γράμματα */
        border-radius: 5px 5px 0px 0px; /* Στρογγυλεμένες γωνίες πάνω */
        margin-right: 5px; /* Κενό ανάμεσα στα tabs */
        padding: 10px 20px;
    }

    /* 2. Στυλ για το ΕΝΕΡΓΟ Tab (Selected) */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #4a90e2 !important; /* Ξεκούραστο Μπλε */
        color: white !important; /* Λευκά γράμματα */
        border: none;
    }

   
    </style>
    """, unsafe_allow_html=True)

    st.title("👁️ Φάση 4: Επεξήγηση Μοντέλου (XAI)")
    st.markdown("---")
    
    st.write("""
    Αναλύσαμε το μοντέλο χρησιμοποιώντας **SHAP** για να κατανοήσουμε τους παράγοντες που οδηγούν σε αθέτηση δανείου. 
    Η παρακάτω ανάλυση βασίζεται στα κύρια χαρακτηριστικά που εντοπίστηκαν: **Υιοθέτηση Τεχνολογίας**, **Έξοδα** και **Τοποθεσία**.
    """)

    # 3 Tabs Layout
    tab1, tab2, tab3 = st.tabs([
        "🌍 Γενική Επισκόπηση", 
        "🔬 Τάσεις Χαρακτηριστικών", 
        "👤 Ανάλυση Πελατών"
    ])

    # --- TAB 1: SUMMARY PLOT ---
    with tab1:
        st.header("Συνολική Σπουδαιότητα Χαρακτηριστικών")
        st.info("Κύριοι Παράγοντες Κινδύνου (κατά σειρά σπουδαιότητας)")
        
        col_img, col_txt = st.columns([2, 1])
        
        with col_img:
            # Displays the Beeswarm plot
            if os.path.exists("shap_summary.png"):
                st.image("shap_summary.png", caption="Σύνοψη SHAP: Η Υιοθέτηση Τεχνολογίας & τα Έξοδα είναι βασικοί παράγοντες", use_container_width=True)
            else:
                st.error("⚠️ Λείπει το αρχείο: 'shap_summary.png'")
        
        with col_txt:
            st.subheader("💡 Βασικά Ευρήματα:")
            st.markdown("""
            1. **Υιοθέτηση Τεχνολογίας (#1):** Παραδόξως, οι **Κόκκινες κουκκίδες** (Χρήστες Τεχνολογίας) βρίσκονται δεξιά, υποδηλώνοντας ότι το μοντέλο θεωρεί την τεχνολογία ως *Παράγοντα Κινδύνου*.
            2. **Υψηλά Έξοδα (#2):** Τα υψηλά μηνιαία έξοδα γενικά αυξάνουν τον κίνδυνο (Κόκκινες κουκκίδες δεξιά).
            3. **Μη Χρήση Κάρτας (#4):** Οι ανενεργοί πελάτες (`No Usage`) επισημαίνονται ως υψηλότερου κινδύνου, επιβεβαιώνοντας τη θεωρία του "Πελάτη Φάντασμα".
            """)

    # --- TAB 2: DEPENDENCE PLOTS ---
    with tab2:
        st.header("Εις Βάθος Ανάλυση: Τάσεις Δεδομένων")
        
        col_dep1, col_dep2 = st.columns(2)
        
        # Plot 1: Tech Adoption Trend
        with col_dep1:
            st.subheader("Υιοθέτηση Τεχνολογίας vs Κίνδυνος")
            if os.path.exists("dependence_plot_tech.png"):
                st.image("dependence_plot_tech.png", caption="Μη Χρήστες Τεχνολογίας (Αριστερά) vs Χρήστες (Δεξιά)", use_container_width=True)
                st.info("""
                **Οπτική Απόδειξη:**
                * **Αριστερή Πλευρά (Μη Χρήστες):** Οι κουκκίδες πέφτουν σημαντικά κάτω από το 0 (-0.10). Αυτό σημαίνει ότι η **μη** χρήση τεχνολογίας μειώνει το σκορ κινδύνου.
                * **Δεξιά Πλευρά (Χρήστες Τεχνολογίας):** Οι κουκκίδες συγκεντρώνονται ψηλότερα, συμβάλλοντας θετικά στον κίνδυνο αθέτησης.
                """)
            else:
                st.warning("Λείπει: 'dependence_plot_tech.png'")

        # Plot 2: High Expenses Trend
        with col_dep2:
            st.subheader("Υψηλά Έξοδα vs Κίνδυνος")
            if os.path.exists("dependence_plot_high_exp.png"):
                st.image("dependence_plot_high_exp.png", caption="Επίπτωση των Υψηλών Εξόδων", use_container_width=True)
                st.info("""
                **Οπτική Απόδειξη:**
                * Στο **X=1 (Υψηλά Έξοδα)**, βλέπουμε μια κάθετη συγκέντρωση.
                * Ενώ ορισμένες κουκκίδες είναι χαμηλά, πολλές ωθούνται προς τα πάνω, επιβεβαιώνοντας το εύρημα της Γενικής Επισκόπησης ότι τα Υψηλά Έξοδα είναι γενικά παράγοντας κινδύνου σε αυτή τη διαμόρφωση του μοντέλου.
                """)
            else:
                st.warning("Λείπει: 'dependence_plot_high_exp.png'")

    # --- TAB 3: WATERFALL PLOTS (LOCAL ANALYSIS) ---
    with tab3:
        st.header("Μελέτη Περίπτωσης: Ασφαλής vs Επικίνδυνος")
        st.write("Σύγκριση δύο συγκεκριμένων πελατών έναντι του **Ορίου (Threshold) 0.40**.")
        
        col_safe, col_risk = st.columns(2)
        
        # --- SAFE CUSTOMER (waterfall 1.png) ---
        with col_safe:
            st.success("✅ Ασφαλής Πελάτης (Πιθανότητα: 0.263)")
            if os.path.exists("waterfall1.png"):
                st.image("waterfall1.png", caption="Πρόβλεψη: 0.263 < 0.40 (Εγκρίθηκε)", use_container_width=True)
                st.markdown("""
                **Γιατί Εγκρίθηκε;**
                * **Η Τοποθεσία τους Έσωσε:** Η μεγάλη Μπλε μπάρα (`district_name_Nymburk`, -0.12) μείωσε δραστικά το σκορ τους.
                * Παρόλο που είχαν **Υψηλά Έξοδα** (+0.02 Κόκκινο) και **Υιοθέτηση Τεχνολογίας** (+0.02 Κόκκινο), ο παράγοντας τοποθεσία ήταν αρκετά ισχυρός για να τους κρατήσει κάτω από το όριο του 0.40.
                """)
            else:
                st.warning("Λείπει: 'waterfall1.png'")

        # --- RISKY CUSTOMER (waterfall2.png) ---
        with col_risk:
            st.error("❌ Πελάτης Υψηλού Κινδύνου (Πιθανότητα: 0.675)")
            if os.path.exists("waterfall2.png"):
                st.image("waterfall2.png", caption="Πρόβλεψη: 0.675 > 0.40 (Απορρίφθηκε)", use_container_width=True)
                st.markdown("""
                **Γιατί Απορρίφθηκε;**
                * **Η Τοποθεσία τους Έβλαψε:** Σε αντίθεση με τον ασφαλή πελάτη, η διαμονή στο `Pribram` (+0.08 Κόκκινο) πρόσθεσε σημαντικό κίνδυνο.
                * **Συσσωρευμένος Κίνδυνος:** Επίσης εμφάνισαν παράγοντες κινδύνου στα **Υψηλά Έξοδα** (+0.03) και την **Υιοθέτηση Τεχνολογίας** (+0.02), ανεβάζοντας τη συνολική πιθανότητα στο **0.675**.
                """)
            else:
                st.warning("Λείπει: 'waterfall2.png'")
# ------------------------------------------------------------------
# PAGE 5: PLAYGROUND (FIXED SCALING & CAPS)
# ------------------------------------------------------------------
elif selection == "5. Risk Playground (Demo)":
    st.title("🎮 Phase 5: Risk Scoring Playground")
    st.markdown("---")
    
    if data is None:
        st.error("⚠️ Το μοντέλο δεν βρέθηκε! Βεβαιώσου ότι το 'final_credit_risk_model.pkl' είναι στον ίδιο φάκελο.")
    else:
        model = data['model']
        
        # Sidebar Inputs
        st.sidebar.markdown("---")
        st.sidebar.header("📝 Νέα Αίτηση Δανείου")
        
        # 1. Δημογραφικά
        age = st.sidebar.slider("Age", 18, 80, 28) # Default 28
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        
        # 2. Τοποθεσία
        region = st.sidebar.selectbox("Region", 
            ["Prague", "central Bohemia", "south Moravia", "north Moravia", "north Bohemia", "east Bohemia", "south Bohemia", "west Bohemia"])
        
        district_name = st.sidebar.selectbox("District Name", 
            ["Hl.m. Praha", "Benesov", "Beroun", "Kladno", "Most", "Karvina", "Ostrava - mesto", "Brno - mesto"])
        
        # 3. Οικονομικά
        loan_amount = st.sidebar.selectbox("Loan Amount", ["low_amount", "mid_amount", "high_amount"])
        loan_duration = st.sidebar.selectbox("Duration", ["short_term", "long_term"])
        monthly_expenses = st.sidebar.selectbox("Monthly Expenses", ["Low_Expenses", "Medium_Expenses", "High_Expenses"])
        
        # 4. Συμπεριφορά
        tech_adoption = st.sidebar.radio("Tech User (App)?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        card_usage = st.sidebar.selectbox("Card Usage", ["No Usage", "Low Usage", "High Usage"])
        
        # Κουμπί Πρόβλεψης
        if st.sidebar.button("Predict Risk"):
            input_data = pd.DataFrame({
                'age': [age], 'gender': [gender], 'region': [region], 'district_name': [district_name],
                'loan_amount': [loan_amount], 'loan_duration': [loan_duration],
                'total_monthly_order': [monthly_expenses], 'tech_adoption_flag': [tech_adoption],
                'card_usage_cat': [card_usage]
            })
            
            try:
                # 1. Base Score από το Μοντέλο
                prob_bad = model.predict_proba(input_data)[:, 0][0] 
                
                # ΑΛΛΑΓΗ: Πολλαπλασιαστής x3 (αντί για x4) για να μην ξεφεύγει
                raw_base = prob_bad * 100 * 3
                
                # ΑΛΛΑΓΗ: Hard Cap στο 95% για να μην βλέπουμε 175%
                base_score = min(raw_base, 95.0) 
                
                # 2. SCORING SYSTEM (PENALTIES & BONUSES)
                penalty_score = 0
                bonus_score = 0 
                
                # --- PENALTIES ---
                if tech_adoption == 0: penalty_score += 15
                if card_usage == "No Usage": penalty_score += 15
                if region in ["north Bohemia", "north Moravia", "Most", "Ostrava - mesto", "Karvina"]: penalty_score += 10
                if monthly_expenses == "High_Expenses": penalty_score += 10
                
                # --- BONUSES ---
                if monthly_expenses == "Low_Expenses": bonus_score += 15
                if tech_adoption == 1: bonus_score += 10
                if loan_duration == "short_term": bonus_score += 5
                if card_usage == "Low Usage":
                    
                    bonus_score += 10 
                elif card_usage == "High Usage":
                    
                    bonus_score += 5

                # 3. Τελικός Υπολογισμός
                final_score = base_score + penalty_score - bonus_score
                
                # Κόφτες Τελικού Σκορ
                if final_score < 1.0: final_score = 1.0
                if final_score > 99.9: final_score = 99.9
                
                st.subheader("📊 Αποτέλεσμα Εκτίμησης")
                
                # Εμφάνιση Scores
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Risk Score", f"{final_score:.1f}%")
                c2.metric("AI Base", f"{base_score:.1f}%", help=f"Raw Prob: {(prob_bad*100):.1f}%")
                c3.metric("Penalties", f"+{penalty_score}%")
                c4.metric("Bonuses", f"-{bonus_score}%", delta_color="normal")
                
                st.progress(int(final_score))
                
                # 4. ΟΡΙΑ & ΖΩΝΕΣ
                if final_score < 40: 
                    st.success("✅ **APPROVED (Green Zone)**")
                    st.write("Ο πελάτης είναι ασφαλής (Low Risk).")
                    st.balloons()
                elif final_score < 75:
                    st.warning("⚠️ **MANUAL REVIEW (Yellow Zone)**")
                    st.write("Μέτριο Ρίσκο. Απαιτείται έλεγχος.")
                else:
                    st.error("🛑 **HIGH RISK (Red Zone)**")
                    st.write("Υψηλό Ρίσκο.")
                
                st.markdown("---")
                col_bad, col_good = st.columns(2)
                
                with col_bad:
                    st.write("**Risk Drivers (Negative):**")
                    if penalty_score > 0:
                        if tech_adoption == 0: st.error("❌ No Tech (+15%)")
                        if card_usage == "No Usage": st.error("❌ No History (+15%)")
                        if region in ["north Bohemia", "north Moravia", "Most", "Ostrava - mesto", "Karvina"]: st.error("⚠️ Bad Region (+10%)")
                        if monthly_expenses == "High_Expenses": st.error("⚠️ High Exp (+10%)")
                    else:
                        st.write("- None")

                with col_good:
                    st.write("**Safety Drivers (Positive):**")
                    if bonus_score > 0:
                        if monthly_expenses == "Low_Expenses": st.success("✅ Low Exp (-15%)")
                        if tech_adoption == 1: st.success("✅ Tech User (-10%)")
                        if loan_duration == "short_term": st.success("✅ Short Term (-5%)")
                        if card_usage == "High Usage": st.success("✅ Active Card (-5%)")
                    else:
                        st.write("- None")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
