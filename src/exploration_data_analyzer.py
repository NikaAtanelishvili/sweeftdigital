import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class ExplorationDataAnalyzer:
    @staticmethod
    def visualize_correlations(df):
        """
            -Visualize relationships between education, income and age \n
            -Visualize relationships between education and age \n
            -Visualize relationships between income and age \n
        """
        df_c = df.copy()

        df_c['Ages'] = 2025 - df_c["Year_Birth"]

        # Education, Income, Age
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_c, x='Ages', y="Income", hue="Education", alpha=0.7, palette="viridis")

        plt.title("Relationship Between Age, Income, and Education", fontweight='bold')
        plt.xlabel("Age")
        plt.ylabel("Income")
        plt.legend(title="Education")
        plt.show()

        #==========================

        # Group by Education and calculate mean income
        avg_income_by_education = df_c.groupby('Education', observed=True)['Income'].mean().reset_index()

        # Create Age groups
        age_bins = pd.cut(df_c['Ages'], bins=range(0, 101, 10), right=False,
                          labels=[f"{i}-{i + 9}" for i in range(0, 100, 10)])
        df_c['Ages_Group'] = age_bins

        # Group by Age group and calculate mean income
        avg_income_by_age = df_c.groupby('Ages_Group', observed=True)['Income'].mean().reset_index()

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

        # 1: Average Income by Education Level
        sns.barplot(ax=axes[0], data=avg_income_by_education, x='Education', y='Income', hue='Education', legend=False, palette='viridis')
        axes[0].set_title("Average Income by Education Level", fontweight='bold')
        axes[0].set_xlabel("Education Level")
        axes[0].set_ylabel("Average Income")
        axes[0].tick_params(axis='x', rotation=45)

        # 2: Average Income by Age Group
        sns.barplot(ax=axes[1], data=avg_income_by_age, x='Ages_Group', y='Income', palette='viridis', hue='Ages_Group', legend=False)
        axes[1].set_title("Average Income by Age Group", fontweight='bold')
        axes[1].set_xlabel("Age Group")
        axes[1].set_ylabel("Average Income")
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_distributions(df):
        """
            Visualize Income, Education and Age distributions
        """
        df_c = df.copy()

        df_c['Ages'] = 2025 - df_c["Year_Birth"]

        # Create the subplots
        plt.figure(figsize=(18, 6))

        # Plot 1: Income Distribution
        plt.subplot(1, 3, 1)
        sns.histplot(df_c['Income'], kde=True, color='skyblue', bins=30)
        plt.title("Income Distribution", fontweight='bold')
        plt.xlabel("Income")
        plt.ylabel("Frequency")

        # Plot 2: Education Distribution
        plt.subplot(1, 3, 2)
        sns.countplot(data=df_c, x='Education', palette='viridis', hue='Education', legend=False)
        plt.title("Education Level Distribution", fontweight='bold')
        plt.xlabel("Education Level")
        plt.ylabel("Count")
        plt.xticks(rotation=45)

        # Plot 3: Age Distribution
        plt.subplot(1, 3, 3)
        sns.histplot(df_c['Ages'], kde=True, color='green', bins=30)
        plt.title("Age Distribution", fontweight='bold')
        plt.xlabel("Age")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_purchases(df):
        df_c = df.copy()

        def calculate_family_size(row):
            # Start with the number of children and teenagers
            family_size = row['Kidhome'] + row['Teenhome']

            # Add parents based on relationship status
            if row['Marital_Status'] in ['Together', 'Married']:
                family_size += 2  # 2 parents
            elif row['Marital_Status'] in ['Single', 'Divorced', 'Widow']:
                family_size += 1  # 1 parent
            elif row['Marital_Status'] in ['Alone']:
                family_size += 0  # 0 parents

            return family_size

        # Apply the function to calculate family size
        df_c['FamilySize'] = df_c.apply(calculate_family_size, axis=1)

        # Set up the figure for multiple subplots
        plt.figure(figsize=(14, 10))

        # Scatter plots for different product spending vs Family Size
        product_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                           'MntGoldProds']
        titles = ['Wines Spending', 'Fruits Spending', 'Meat Products Spending', 'Fish Products Spending',
                  'Sweet Products Spending', 'Gold Products Spending']

        for i, product in enumerate(product_columns):
            plt.subplot(2, 3, i + 1)
            sns.scatterplot(data=df_c, x='FamilySize', y=product, alpha=0.6, color='teal')
            plt.title(f"Family Size vs {titles[i]}")
            plt.xlabel("Family Size")
            plt.ylabel(f"Spending on {titles[i]}")

        # Adjust layout
        plt.tight_layout()
        plt.show()

        #==============================================================

        # Calculate total purchases for each category
        total_web_purchases = df_c['NumWebPurchases'].sum()
        total_store_purchases = df_c['NumStorePurchases'].sum()

        # Categorize visit frequency
        df_c['Visit_Group'] = pd.cut(df_c['NumWebVisitsMonth'], bins=[0, 2, 5, 10, 20, 50],
                                   labels=['0-2', '3-5', '6-10', '11-20', '21+'])

        # Calculate conversion rate per group
        conversion_rate = df_c.groupby('Visit_Group', observed=True)['NumWebPurchases'].sum() / df_c.groupby('Visit_Group', observed=True).size()
        conversion_rate = conversion_rate.reset_index(name='Conversion Rate')

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart for Web Purchases vs Store Purchases
        axes[0].pie([total_web_purchases, total_store_purchases], labels=['Web Purchases', 'Store Purchases'],
                    autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
        axes[0].set_title("Total Web Purchases vs Store Purchases")

        # Bar plot for conversion rate by web visit frequency
        sns.barplot(data=conversion_rate, x='Visit_Group', y='Conversion Rate', palette='viridis', hue='Visit_Group', legend=False, ax=axes[1])
        axes[1].set_title("Conversion Rate by Web Visit Frequency", fontweight="bold")
        axes[1].set_xlabel("Number of Visits per Month")
        axes[1].set_ylabel("Purchase Conversion Rate")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_relationships(df):
        df_c = df.copy()
        
        df_c['Marital_Status'] = df_c['Marital_Status'].astype('category')

        # Group by Marital_Status and compute average Kidhome and Teenhome
        grouped_df = df_c.groupby('Marital_Status', as_index=False, observed=False).agg({
            'Kidhome': 'mean',
            'Teenhome': 'mean'
        })

        # 2. Melt the dataframe for grouped bar chart
        melted_df = grouped_df.melt(
            id_vars='Marital_Status',
            value_vars=['Kidhome', 'Teenhome'],
            var_name='Children_Type',
            value_name='Average_Count'
        )

        # 3. Compute frequency of each marital status
        marital_status_counts = df_c['Marital_Status'].value_counts().reset_index()
        marital_status_counts.columns = ['Marital_Status', 'Count']

        # Create a figure with two subplots (side by side)
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))

        # 1: Average Number of Children and Teenagers by Marital Status
        sns.barplot(
            data=melted_df,
            x='Marital_Status',
            y='Average_Count',
            hue='Children_Type',
            palette='viridis',
            ax=ax[0]
        )
        ax[0].set_title("Average Number of Children and Teenagers by Marital Status", fontweight='bold')
        ax[0].set_xlabel("Marital Status", fontweight='bold')
        ax[0].set_ylabel("Average Count", fontweight='bold')
        ax[0].set_xticks(range(len(ax[0].get_xticklabels())))  # Set tick positions
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
        ax[0].legend(title="Type", loc="upper right")

        # 2: Marital Status Distribution
        sns.barplot(
            data=marital_status_counts,
            x='Marital_Status',
            y='Count',
            palette='OrRd',
            hue='Marital_Status',
            legend=False,
            ax=ax[1]
        )
        ax[1].set_title("Marital Status Distribution", fontweight='bold')
        ax[1].set_xlabel("Marital Status", fontweight='bold')
        ax[1].set_ylabel("Frequency", fontweight='bold')
        ax[1].set_xticks(range(len(ax[1].get_xticklabels())))
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()
