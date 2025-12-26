from sqlalchemy import create_engine, text
import pandas as pd

class AnalyticsDashboard:
    def __init__(self, connection_string):
        """
        Initializes the engine using a SQLAlchemy connection string.
        Format: postgresql://user:password@host:port/database
        """
        self.engine = create_engine(connection_string)

    def _execute_query(self, query):
        """Helper to run query using SQLAlchemy and return headers + results."""
        try:
            with self.engine.connect() as conn:
             
                result = conn.execute(text(query))
                
               
                headers = result.keys()
                
           
                data = result.fetchall()
                
                return list(headers), data
        except Exception as e:
            return None, str(e)

    # 1. Overall Default Rate
    def get_overall_default_rate(self):
        query = """
        SELECT 
            COUNT(*) AS total_loans,
            SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END) AS bad_loans,
            ROUND((SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)) * 100, 2) AS default_rate
        FROM loan;
        """
        return self._execute_query(query)

    # 2. Correlation: District Unemployment vs Defaults
    def get_unemployment_correlation(self):
        query = """
        WITH corr_data AS(
            SELECT d."District_name" AS District,
                   d."Unemployment_rate_95" AS Unemployment,
                   COUNT(l.loan_id) Total_loans,
                   SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END) AS bad_loans,
                   ROUND((SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)) * 100, 2) AS Default_rate       
            FROM loan AS l 
            JOIN account AS a ON l.account_id = a.account_id
            JOIN district AS d ON d."District_code" = a.district_id           
            GROUP BY d."District_name", "Unemployment_rate_95"
            HAVING COUNT(l.loan_id) > 10
            ORDER BY d."Unemployment_rate_95" DESC
        )
        SELECT ROUND(CORR(Default_rate, Unemployment)::NUMERIC, 3) AS Correlation
        FROM corr_data;
        """
        return self._execute_query(query)

    # 3. Correlation: Loan Duration vs Default Rate
    def get_duration_correlation(self):
        query = """
        WITH data_loan_corr AS(
            SELECT duration AS duration,
                   COUNT(*) AS total_loans,
                   SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END) AS bad_loans,
                   ROUND((SUM(CASE WHEN status IN ('B', 'D') THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)) * 100, 2) AS Default_rate
            FROM loan
            GROUP BY duration
            ORDER BY duration ASC
        )
        SELECT ROUND(CORR(Duration, Default_rate)::NUMERIC, 2) * 100 AS correlation
        FROM data_loan_corr;
        """
        return self._execute_query(query)

    # 4. Client Demographics (Age Group & Gender)
    def get_client_demographics(self):
        query = """
        SELECT     
             CASE 
                WHEN (1999 - (1900 + SUBSTRING(birth_number::text, 1, 2)::int)) < 25 THEN 'Under 25'
                WHEN (1999 - (1900 + SUBSTRING(birth_number::text, 1, 2)::int)) BETWEEN 25 AND 40 THEN '25-40'
                WHEN (1999 - (1900 + SUBSTRING(birth_number::text, 1, 2)::int)) BETWEEN 41 AND 55 THEN '41-55'
                WHEN (1999 - (1900 + SUBSTRING(birth_number::text, 1, 2)::int)) > 55 THEN 'Over 55'
             END AS age_group,
             SUM(CASE WHEN gender = 'Male' THEN 1 ELSE 0 END) AS Num_males,
             SUM(CASE WHEN gender = 'Female'THEN 1 ELSE 0 END) AS Num_females,
             COUNT(*) AS total_clients
        FROM client
        GROUP BY age_group
        ORDER BY age_group ASC;
        """
        return self._execute_query(query)

    # 5. VIP Clients (Gold Cards by Region)
    def get_vip_clients(self):
        query = """
        SELECT dis."Region" AS Region,
               dis."District_name" AS District,
               COUNT(c.card_id) AS Gold_cards
        FROM card AS c 
        JOIN disp AS d ON c.disp_id = d.disp_id
        JOIN client AS cl ON cl.client_id = d.client_id
        JOIN district AS dis ON dis."District_code" = cl.district_id
        WHERE c.type = 'gold'
        GROUP BY dis."Region", dis."District_name"
        ORDER BY COUNT(c.card_id) DESC;
        """
        return self._execute_query(query)

    # 6. Average Age Joined by Year
    def get_avg_age_joined(self):
        query = """
        WITH client_first_acc AS (
            SELECT c.client_id AS Client_id,
                   c.birth_year AS Birth_year,
                   MIN(acc_date) AS first_acc
            FROM client AS c 
            JOIN disp d ON c.client_id= d.client_id
            JOIN account a ON a.account_id = d.account_id
            GROUP BY c.client_id, c.birth_year
        )
        SELECT EXTRACT(YEAR FROM first_acc) AS join_year,
               ROUND(AVG(EXTRACT(YEAR FROM first_acc) - Birth_year), 1) AS avg_age_joined
        FROM client_first_acc
        GROUP BY join_year
        ORDER BY join_year ASC;
        """
        return self._execute_query(query)

    # 7. Min Balance: Defaulters vs Non-Defaulters
    def get_min_balance_comparison(self):
        query = """
        SELECT 
            CASE 
                WHEN l.status IN ('B', 'D') THEN 'Defaulter (Bad Loan)'
                WHEN l.status IN ('A', 'C') THEN 'Non-Defaulter (Good Loan)'
            END AS loan_category,
            ROUND(AVG(min_bal)::numeric, 2) AS avg_minimum_balance_ever
        FROM loan l
        JOIN (
            SELECT account_id, MIN(balance) as min_bal 
            FROM trans 
            GROUP BY account_id
        ) t ON l.account_id = t.account_id
        GROUP BY loan_category
        ORDER BY avg_minimum_balance_ever DESC;
        """
        return self._execute_query(query)

    # 8. Transaction Ratio (Cash vs Card)
    def get_transaction_ratios(self):
        query = """
        SELECT 
            SUM(CASE WHEN operation = 'Cash_Withdrawal' THEN 1 ELSE 0 END) AS cash_withdrawal_count,
            SUM(CASE WHEN operation = 'Credit_Card_Withdrawal' THEN 1 ELSE 0 END) AS card_withdrawal_count,
            ROUND(
                SUM(CASE WHEN operation = 'Cash_Withdrawal' THEN 1.0 ELSE 0 END) / 
                NULLIF(SUM(CASE WHEN operation = 'Credit_Card_Withdrawal' THEN 1.0 ELSE 0 END), 0), 
                1
            ) AS transaction_ratio_cash_to_card,
            ROUND(SUM(CASE WHEN operation = 'Cash_Withdrawal' THEN amount ELSE 0 END)::NUMERIC,0) AS cash_withdrawal_amount,
            ROUND(SUM(CASE WHEN operation = 'Credit_Card_Withdrawal' THEN amount ELSE 0 END)::NUMERIC,0) AS card_withdrawal_amount
        FROM trans;
        """
        return self._execute_query(query)

    # 9. Sanction Interest (Fines Analysis)
    def get_sanction_interest(self):
        query = """
        SELECT
             CASE 
                WHEN l.status IN ('B', 'D') THEN 'Defaulter (Bad Loan)'
                WHEN l.status IN ('A', 'C') THEN 'Non-Defaulter (Good Loan)'
             END AS client_category,
            COUNT(DISTINCT l.loan_id) AS Total_clients,
            ROUND(COUNT(DISTINCT CASE WHEN t.k_symbol = 'Sanction_Interest' THEN l.loan_id END):: NUMERIC /
            COUNT(DISTINCT l.loan_id) * 100, 2) AS per_with_fines
        FROM trans AS t 
        JOIN loan AS l ON t.account_id = l.account_id
        GROUP BY client_category;
        """
        return self._execute_query(query)