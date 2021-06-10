##############################################################################
# Understanding the impact of Growth and Margin profile on B2B SaaS Valuations
# Dataset: 30 High growth B2B SaaS companies
#   Author: Ramu Arunachalam (ramu@acapital.com)
#   Created: 06/09/21
#   Datset last updated: 06/08/21
###############################################################################

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import lux
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt


def load_dataset(filename = '2021-06-08-comps.csv'):
    df_main = pd.read_csv(filename)

    # Cleanup
    df_obj = df_main.select_dtypes(['object'])
    df_main[df_obj.columns] = df_obj \
        .apply(lambda x: x.str.strip('x')) \
        .apply(lambda x: x.str.strip('%')) \
        .replace(',', '', regex=True) \
        .replace('\$', '', regex=True)
    return df_main

def augment_datset(df):
    df['2021 Revenue Growth'] = (df['2021 Analyst Revenue Estimates'].astype(float) / df['2020 Revenue'].astype(float) - 1) * 100
    return df


def get_scatter_fig(df, x, y):
    fig = px.scatter(df,
                     x=x,
                     y=y,
                     hover_data=['Name'],
                     title=f'{y} vs {x}')
    regline = sm.OLS(df[y], sm.add_constant(df[x])).fit().fittedvalues
    fig.add_trace(go.Scatter(x=df[x],
                             y=regline,
                             mode='lines',
                             marker_color='black',
                             name='Best-fit',
                             line=dict(width=4, dash='dot')))
    return fig


latex_dict = {'EV / NTM Revenue':r'''\frac{EV}{Rev_{NTM}}''',
              'EV / 2021 Revenue':r'''\frac{EV}{Rev_{2021}}''',
              'NTM Revenue Growth':r'''Rev\,Growth_{NTM}''',
              '2021 Revenue Growth':r'''Rev\,Growth_{2021}''',
              'Gross Margin':r'Gross Margin',
              'Growth adjusted EV / NTM Revenue':r'''\frac{EV}{Rev_{NTM}}\cdot\frac{1}{Growth_{NTM}}''',
              'Growth adjusted EV / 2021 Revenue':r'''\frac{EV}{Rev_{2021}}\cdot\frac{1}{Growth_{2021}}'''
             }

def regression(df,x_cols,reg_y):
    if not x_cols:
        return

    # Print regression equation
    lstr = latex_dict[reg_y] + r'''= \beta_0'''
    for x, i in zip(x_cols, range(len(x_cols))):
        lstr += rf'''+\beta_{{{i + 1}}}\cdot {{{latex_dict[x]}}}'''
    lstr += r'''+\epsilon'''
    st.latex(lstr)

    X = df[x_cols]
    X = sm.add_constant(X)
    model = sm.OLS(df[reg_y], X).fit()

    #### Print p-values
    df_pvalues = model.pvalues \
        .to_frame() \
        .reset_index() \
        .rename(columns={'index': 'vars', 0: 'p-value'})

    df_pvalues['Statistical Significance'] = 'Low'
    df_pvalues.loc[df_pvalues['p-value'] <= 0.05, 'Statistical Significance'] = 'High'
    df_pvalues = df_pvalues[df_pvalues['vars'] != 'const']

    def highlight_significant_rows(val):
        color = 'green' if val['p-value'] <= 0.05 else 'red'
        return [f"color: {color}"] * len(val)

    st.table(df_pvalues.style.apply(highlight_significant_rows, axis=1))
    st.write("** Summary: **")
    if model.f_pvalue < 0.05:
        st.write(f"* The regression is **statistically significant** (F-test = {model.f_pvalue:.2f})")
    else:
        st.write(f"* The regression is **NOT statistically significant** (F-test = {model.f_pvalue:.2f})")
    st.write(f"* R^2 = {model.rsquared * 100:.2f}% ")

    for _, row in df_pvalues.iterrows():
        str = 'strong' if row['Statistical Significance'] == 'High' else 'poor'
        st.write(f"* There is a **{str} relationship** between *'{reg_y}'* and *'{row['vars']}'*")

    ##

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

    option = st.sidebar.selectbox('Timeline', ('NTM', '2021'))

    # Pick metrics
    gm = 'Gross Margin'
    rev_g = f'{option} Revenue Growth'
    rev_mult = f'EV / {option} Revenue'
    growth_adj_mult = f'Growth adjusted {rev_mult}'

    df = df_main[['Name', rev_mult, gm, rev_g]].dropna()
    df[[rev_mult, gm, rev_g]] = df[[rev_mult, gm, rev_g]].astype(float)
    df[growth_adj_mult] = df[rev_mult] / df[rev_g]

    y_sel = st.sidebar.radio("Target metric", [rev_mult, growth_adj_mult, rev_g])


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
    ## Plots
    st.sidebar.info('Plots:')
    st.subheader("**Plots**")

    st.sidebar.text('Plot X-Axis')
    if st.sidebar.checkbox(f'{rev_g}', value=True):
        st.plotly_chart(get_scatter_fig(df, x=rev_g, y=y_sel))

    if st.sidebar.checkbox(f'Gross Margin', value=True):
        st.plotly_chart(get_scatter_fig(df, x=gm, y=y_sel))


    ## Regression
    st.sidebar.info("Regression:")
    st.subheader("Regression")
    st.sidebar.text("Select independent variable(s)")
    regress_growth = st.sidebar.checkbox(rev_g, value=True, key='reg_rev_g')
    regress_gross_margin = st.sidebar.checkbox(gm, value=True, key='reg_gm')

    x_cols = [rev_g] if regress_growth else []
    x_cols += [gm] if regress_gross_margin else x_cols
    regression(df, x_cols=x_cols,reg_y=y_sel)

    st.subheader("Dataset")
    st.beta_expander('Table Output').table(df)
    st.beta_expander('Full Raw Table Output').table(df_main)

    return


if __name__ == "__main__":
    main()

