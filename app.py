##############################################################################
# Understanding the impact of Growth and Margin profile on B2B SaaS Valuations
# Dataset: 30 High growth B2B SaaS companies
#   Author: Ramu Arunachalam (ramu@acapital.com)
#   Created: 06/09/21
#   Datset last updated: 06/08/21
###############################################################################

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

latest_filename_all = '2021-06-11-comps_B2B_ALL.csv'
latest_filename_high_growth = '2021-06-11-comps_B2B_High_Growth.csv'


def load_dataset(f_all=latest_filename_all, f_hg=latest_filename_high_growth):
    df_main = pd.read_csv(f_all)
    df_main_hg = pd.read_csv(f_hg)
    tickers_all = list(df_main[df_main['Name'].isin(['Median', 'Mean']) == False]['Name'])
    tickers_hg = list(df_main_hg[df_main_hg['Name'].isin(['Median', 'Mean']) == False]['Name'])
    tickers_excl_hg = list(set(tickers_all) - set(tickers_hg))

    sel = st.sidebar.radio("Dataset", ['All B2B SaaS', 'Only High Growth', 'Exclude High Growth'])
    if sel == 'Only High Growth':
        df_main = df_main[df_main['Name'].isin(tickers_hg)]
    elif sel == 'Exclude High Growth':
        df_main = df_main[df_main['Name'].isin(tickers_excl_hg)]

    # Cleanze
    df_obj = df_main.select_dtypes(['object'])
    df_main[df_obj.columns] = df_obj \
        .apply(lambda x: x.str.strip('x')) \
        .apply(lambda x: x.str.strip('%')) \
        .replace(',', '', regex=True) \
        .replace('\$', '', regex=True)

    cols = df_main.columns
    for c in cols:
        try:
            df_main[c] = pd.to_numeric(df_main[c])
        except:
            pass

    return df_main


def augment_datset(df):
    df['2021 Revenue Growth'] = (df['2021 Analyst Revenue Estimates'].astype(float) / df['2020 Revenue'].astype(
        float) - 1) * 100
    return df


def get_scatter_fig(df, x, y):
    fig = px.scatter(df,
                     x=x,
                     y=y,
                     hover_data=['Name'],
                     title=f'{y} vs {x}')
    df_r = df[[y] + [x]].dropna()
    regline = sm.OLS(df_r[y], sm.add_constant(df_r[x])).fit().fittedvalues
    fig.add_trace(go.Scatter(x=df_r[x],
                             y=regline,
                             mode='lines',
                             marker_color='black',
                             name='Best-fit',
                             line=dict(width=4, dash='dot')))
    return fig


latex_dict = {'EV / NTM Revenue': r'''\frac{EV}{Rev_{NTM}}''',
              'EV / 2021 Revenue': r'''\frac{EV}{Rev_{2021}}''',
              'EV / NTM Gross Profit': r'''\frac{EV}{GP_{NTM}}''',
              'EV / 2021 Gross Profit': r'''\frac{EV}{GP_{2021}}''',
              'NTM Revenue Growth': r'''Rev\,Growth_{NTM}''',
              '2021 Revenue Growth': r'''Rev\,Growth_{2021}''',
              'Growth adjusted EV / NTM Revenue': r'''\frac{EV}{Rev_{NTM}}\cdot\frac{1}{Growth_{NTM}}''',
              'Growth adjusted EV / 2021 Revenue': r'''\frac{EV}{Rev_{2021}}\cdot\frac{1}{Growth_{2021}}'''
              }


def regression(df_input, x_cols, reg_y):
    if not x_cols:
        return
    df_r = df_input[[reg_y] + x_cols].dropna()

    # Print regression equation
    lstr = latex_dict.get(reg_y, reg_y) + r'''= \beta_0'''
    for x, i in zip(x_cols, range(len(x_cols))):
        lstr += rf'''+\beta_{{{i + 1}}}\cdot {{{latex_dict.get(x, x)}}}'''
    lstr += r'''+\epsilon'''
    st.latex(lstr.replace("%", "\%").replace("&", "\&").replace("$", "\$"))

    X = df_r[x_cols]
    X = sm.add_constant(X)
    model = sm.OLS(df_r[reg_y], X).fit()

    #### Print p-values
    df_pvalues = model.pvalues \
        .to_frame() \
        .reset_index() \
        .rename(columns={'index': 'vars', 0: 'p-value'})
    df_pvalues['Beta'] = model.params.to_frame().reset_index().rename(columns={0: 'Beta'})['Beta']
    df_pvalues['Statistical Significance'] = 'Low'
    df_pvalues.loc[df_pvalues['p-value'] <= 0.05, 'Statistical Significance'] = 'High'
    df_pvalues = df_pvalues[df_pvalues['vars'] != 'const']

    def highlight_significant_rows(val):
        color = 'green' if val['p-value'] <= 0.05 else 'red'
        return [f"color: {color}"] * len(val)

    st.write("** Summary: **")
    st.write(f"* N = {len(df_r)} companies")

    if model.rsquared * 100 > 30:
        st.write(f"* Model fit is **good** R ^ 2 = {model.rsquared * 100: .2f}%")
        if model.f_pvalue < 0.05:
            st.write(f"* Model is **statistically significant** (F-test = {model.f_pvalue:.2f})")
        else:
            st.write(f"* The regression is **NOT statistically significant** (F-test = {model.f_pvalue:.2f})")

        for _, row in df_pvalues.iterrows():
            str = 'strong' if row['Statistical Significance'] == 'High' else 'weak'
            st.write(f"* There is a **{str} relationship** between *'{reg_y}'* and *'{row['vars']}'*")
        st.table(df_pvalues.set_index('vars').style.apply(highlight_significant_rows, axis=1))

    else:
        st.write(f"* Model fit is **poor** (R ^ 2 = {model.rsquared * 100: .2f}%)")

    ##

    # Check for multicolinearity
    # VIF dataframe
    vif_data = pd.DataFrame()
    df_v = df_r[x_cols]
    vif_data["feature"] = df_v.columns

    if len(df_v.columns) > 1:
        # calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(df_v.values, i)
                           for i in range(len(df_v.columns))]
        if len(vif_data[vif_data['VIF'] > 10]) > 0:
            st.write("* **Potential multicolinearity**")
        else:
            st.write("* **NO multicolinearity**")
        st.beta_expander("VIF").table(vif_data.set_index('feature'))

    # Print out the statistics
    st.beta_expander("Regression output").write(model.summary())
    res_ex = st.beta_expander("Regression Residuals")
    # Plot residuals
    for x in x_cols:
        res_ex.write(f"Plotting residuals for **{x}**")
        fig = plt.figure(figsize=(8, 6))
        sm.graphics.plot_regress_exog(model, x, fig=fig)
        res_ex.pyplot(fig)


def main():
    df_main = load_dataset()
    df_main = augment_datset(df_main)

    option = st.sidebar.selectbox('Timeline', ('2021', 'NTM'))

    # Pick metrics
    gm = 'Gross Margin'
    rev_g = f'{option} Revenue Growth'
    rev_mult = f'EV / {option} Revenue'
    growth_adj_mult = f'Growth adjusted {rev_mult}'
    gp_mult = f'EV / {option} Gross Profit'

    df = df_main
    df[growth_adj_mult] = df[rev_mult] / df[rev_g]

    y_sel = st.sidebar.radio("Target metric", [rev_mult, gp_mult, growth_adj_mult])

    st.title('**Impact of Growth and Margins on Valuation**')
    st.info(f'Num companies = {len(df)}')
    """
    **TL;DR**
    
    For high-growth B2B SaaS companies:
    * **Growth drives revenue multiples**
        * Higher growth &#8594 Premium revenue multiples
        * Lower growth &#8594 Discounted revenue multiples
        
    
    * **Gross Margin doesn't impact valuation**
        * All things being equal, companies with higher gross margins don't command higher revenue multiples
        * And conversely, lower gross margins don't seem to pull down revenue multiples
    """

    plot_container = st.beta_container()

    ## Regression
    st.sidebar.info("Regression:")
    st.subheader("Regression")
    st.sidebar.text("Select independent variable(s)")

    reg_x_dict = dict({rev_g: True, gm: True})  # Default ON
    # Check if user selected revenue growth and/or gross margin
    reg_x_cols = [i for i in [rev_g, gm] if st.sidebar.checkbox(i, value=reg_x_dict.get(i, False), key=i)]
    remaining_cols = list(set(df.select_dtypes(['float', 'int']).columns) - set(reg_x_cols))
    reg_x_cols += st.sidebar.multiselect("Additional independent variables:", remaining_cols)
    regression(df, x_cols=reg_x_cols, reg_y=y_sel)

    with plot_container:
        ## Plots
        st.subheader("**Plots**")
        for _, x in zip(range(4), reg_x_cols):
            st.plotly_chart(get_scatter_fig(df, x=x, y=y_sel))
        st.plotly_chart(get_scatter_fig(df, x=gm, y=rev_g))

    st.subheader("Dataset")
    st.beta_expander('Table Output') \
        .table(df[['Name'] + [y_sel] + reg_x_cols]
               .set_index('Name')
               .sort_values(y_sel, ascending=False))
    st.beta_expander('Full Raw Table Output').table(df_main)

    return


if __name__ == "__main__":
    main()
