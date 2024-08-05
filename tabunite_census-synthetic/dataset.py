import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split


class SynthDataset():
    def __init__(self, dataname):
        self.dataname = dataname
        self.data_path = 'data/census.csv' # change to census.csv path
        self.scaler = StandardScaler()
        self.quantile_scaler = None
        self.num_size = 0
        self.data, self.col_names, self.data_cat, self.data_cont = self.load_data()
        self.total_features = self.data.shape[1]
        self.current_index = 0
        self.gen_data = self.generate_data()
        self.shuffled_indices = np.random.permutation(len(self.gen_data))

    def load_data(self):
        census_data = pd.read_csv(self.data_path, header=0)

        # Dropping Columns
        columns_to_drop = ['dIncome2', 'dIncome3', 'dIncome4', 'dIncome5', 'dIncome6', 'dIncome7', 'dIncome8']
        census_data.drop(columns=columns_to_drop, inplace=True)
        
        # Part 1: Categorical to Continuous (Ordinal Features)
        ## Creating Columns
        
        ### Income and Earnings
        dIncome1_mapping = {
            0: 0,
            1: 5000,
            2: 15000,
            3: 35000,
            4: 70000
        }
        
        dRearning_mapping = {
            0: 0,
            1: 5000,
            2: 15000,
            3: 35000,
            4: 70000,
            5: 150000
        }
        
        census_data['dRearning_cont'] = census_data['dRearning'].map(dRearning_mapping) + np.random.uniform(low=0.0, high=1.0, size=census_data.shape[0])
        census_data['dIncome1_cont'] = census_data['dIncome1'].map(dIncome1_mapping) + np.random.uniform(low=0.0, high=1.0, size=census_data.shape[0])
        
        census_data.drop(columns=['dRearning', 'dIncome1'], inplace=True)
        
        ### Age
        age_ranges = {
            0: (0, 9),    # Ages 0-9
            1: (10, 19),  # Ages 10-19
            2: (20, 29),  # Ages 20-29
            3: (30, 39),  # Ages 30-39
            4: (40, 49),  # Ages 40-49
            5: (50, 59),  # Ages 50-59
            6: (60, 69),  # Ages 60-69
            7: (70, 79)   # Ages 70-79
        }

        # Function to randomly sample an age within the given range
        def sample_age(category):
            return np.random.uniform(age_ranges[category][0], age_ranges[category][1])

        # Apply the mapping to the dAge column to create the new continuous age column
        census_data['dAge_cont'] = census_data['dAge'].apply(sample_age) + np.random.uniform(low=0.0, high=1.0, size=census_data.shape[0])

        # Drop the original dAge column
        census_data.drop(columns=['dAge'], inplace=True)
        
        ### English Proficiency
        noise = np.random.uniform(low=0.0, high=0.1, size=census_data.shape[0])
        census_data['iEnglish_cont'] = census_data['iEnglish'] + noise
        census_data.drop(columns=['iEnglish'], inplace=True)
        
        ### Hours
        hours_mapping = {
            0: 0,
            1: 10,
            2: 20,
            3: 30,
            4: 40,
            5: 50
        }
        
        census_data['dHour89_cont'] = census_data['dHour89'].map(hours_mapping) + np.random.uniform(low=0.0, high=1.0, size=census_data.shape[0])
        census_data['dHours_cont'] = census_data['dHours'].map(hours_mapping) + np.random.uniform(low=0.0, high=1.0, size=census_data.shape[0])

        # Drop the original columns
        census_data.drop(columns=['dHour89', 'dHours'], inplace=True)
        
        ### Travel Time
        dTravtime_mapping = {
            0: 0,
            1: 10,
            2: 20,
            3: 30,
            4: 40,
            5: 50,
            6: 60
        }
        
        census_data['dTravtime_cont'] = census_data['dTravtime'].map(dTravtime_mapping) + np.random.uniform(low=0.0, high=1.0, size=census_data.shape[0])
        census_data.drop(columns=['dTravtime'], inplace=True)
        
        ### Schooling and Working Years
        census_data['iYearsch_cont'] = census_data['iYearsch'] + np.random.uniform(low=0.0, high=1.0, size=census_data.shape[0])
        census_data['iYearwrk_cont'] = census_data['iYearwrk'] + np.random.uniform(low=0.0, high=1.0, size=census_data.shape[0])
        census_data.drop(columns=['iYearsch', 'iYearwrk'], inplace=True)
        
        # Part 2: Categorical to Continuous (Non-Ordinal Features Frequency Encoding)
        features_to_encode = [
            'dAncstry1', 'dAncstry2', 'iCitizen', 'iMarital', 
            'dHispanic', 'iClass', 'dPOB', 'dOccup', 
            'dIndustry', 'iMobility', 'iRelat1', 'iSex'
        ]
        
        # Frequency encoding function
        # Can potentially add noise to the frequency encoding
        def frequency_encode(df, columns):
            new_columns = []
            for col in columns:
                freq = df[col].value_counts(normalize=True)
                new_col_name = col + '_cont'
                df[new_col_name] = df[col].map(freq) + np.random.normal(0, 0.1, df.shape[0])
                new_columns.append(new_col_name)
            return df, new_columns
        
        census_data, encoded_columns = frequency_encode(census_data, features_to_encode)
        census_data.drop(columns=features_to_encode, inplace=True)
        
        ## Adding Columns to Dataset
        columns_to_insert = encoded_columns + [
            'iYearsch_cont', 'iYearwrk_cont', 'dTravtime_cont', 'dHour89_cont',
            'dHours_cont', 'iEnglish_cont', 'dAge_cont', 'dRearning_cont', 'dIncome1_cont'
        ]
        
        self.num_size += len(columns_to_insert)
        cols = census_data.columns.tolist()
        for col in columns_to_insert:
            cols.insert(0, cols.pop(cols.index(col)))
        census_data = census_data[cols]

        # Saving column names
        column_names = census_data.columns.to_list()
            
        self.quantile_scaler = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(census_data.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=0,
        )

        # Apply Quantile Transformer to all columns before self.num_size index
        census_data.iloc[:, :self.num_size] = self.quantile_scaler.fit_transform(census_data.iloc[:, :self.num_size])
        
        # cont. + cat. data, col names, categorical data, continuous data
        return census_data.to_numpy(), column_names, census_data.iloc[:, self.num_size:].to_numpy(), census_data.iloc[:, :self.num_size].to_numpy()

    def generate_random_feature_indices(self, num_indices, exclude, seed):
        np.random.seed(seed)
        start, end = 0, self.total_features
        possible_indices = np.arange(start, end)
        filtered_indices = [idx for idx in possible_indices if idx not in exclude]
        indices = np.random.choice(filtered_indices, size=num_indices, replace=False)
        return np.sort(indices)

    def basic_label_generation(self, syn, syn_type, seed, rand):
        X = self.data_cont
        X = self.scaler.fit_transform(X)  # Model as a Gaussian

        # if rand == False: # columns that are not random and makes logical sense
        if syn_type == 'syn1':
            logit = np.exp(X[:, syn[0]] * X[:, syn[1]])

        elif syn_type == 'syn2':
            logit = np.exp(np.sum(X[:, syn]**2, axis=1) - 4.0)
            
        else: # syn_type == 'syn3'
            logit = -10 * np.sin(0.2 * X[:, syn[0]]) + \
                    np.abs(X[:, syn[1]] + X[:, syn[2]]) + \
                    np.exp(-X[:, syn[3]]) - 2.4

        y = 1 / (1 + logit)  # Logistic transformation
        y += np.random.normal(0, 0.1, len(y))
        return y.reshape(-1, 1)

    def generate_data(self):
        # Generating synthetic data columns
        synthetic_data = []
        col_names = []
        
        # syn1 (Combination of 2 Features)
        syn1 = [
            [3, 4], # dHours_cont and dHour89_cont - Both are related to hours worked.
            [1, 7], # dAge_cont and iYearsch_cont - Age and years of schooling might show educational attainment with age.
            [2, 13], # iEnglish_cont and dPOB_cont - English proficiency and place of birth might relate to language skills based on birthplace.
            [17, 15], # iCitizen_cont and dHispanic_cont - Citizenship status and Hispanic origin might indicate demographic relationships.
            [5, 6], # dTravtime_cont and iYearwrk_cont - Travel time to work and years worked might indicate commuting patterns.
            [16, 9], # iMarital_cont and iRelat1_cont - Marital status and relationship to household head might reflect family structure.
            [11, 12], # dIndustry_cont and dOccup_cont - Industry and occupation are directly related to employment.
            [8, 0], # iSex_cont and dRearning_cont - Gender and earnings might show income disparities.
            [10, 17], # iMobility_cont and iCitizen_cont - Mobility status and citizenship might indicate migration patterns.
            [19, 18] # dAncstry1_cont and dAncstry2_cont - Ancestry from both parents might give a fuller picture of heritage.
        ]

        # syn2 (Combination of 3 Features)
        syn2 = [
            [3, 4, 5], # dHours_cont, dHour89_cont, and dTravtime_cont - Hours worked, hours worked in 1989, and travel time to work might indicate work-life balance.
            [1, 7, 6], # dAge_cont, iYearsch_cont, and iYearwrk_cont - Age, years of schooling, and years worked can indicate career progression.
            [2, 13, 17], # iEnglish_cont, dPOB_cont, and iCitizen_cont - English proficiency, place of birth, and citizenship might relate to assimilation.
            [16, 9, 10], # iMarital_cont, iRelat1_cont, and iMobility_cont - Marital status, relationship to household head, and mobility status might reflect household dynamics.
            [11, 12, 0], # dIndustry_cont, dOccup_cont, and dRearning_cont - Industry, occupation, and earnings are closely related to job characteristics.
            [8, 0, 1], # iSex_cont, dRearning_cont, and dAge_cont - Gender, earnings, and age might show income trends across genders and ages.
            [19, 18, 17], # dAncstry1_cont, dAncstry2_cont, and iCitizen_cont - Ancestry and citizenship might indicate heritage and immigration status.
            [15, 9, 1], # dHispanic_cont, iRelat1_cont, and dAge_cont - Hispanic origin, relationship to household head, and age might reflect demographic patterns.
            [5, 6, 12], # dTravtime_cont, iYearwrk_cont, and dOccup_cont - Travel time, years worked, and occupation can relate to job location and stability.
            [7, 6, 0] # iYearsch_cont, iYearwrk_cont, and dRearning_cont - Years of schooling, years worked, and earnings can indicate education’s impact on income.
        ]
        
        syn_dict = {'syn1': syn1, 'syn2': syn2}

        for syn_type, syn in syn_dict.items():
            for i in range(len(syn)):
                print(f"Generating cont. predefined column {i} for polynomial {syn_type}")
                Y = self.basic_label_generation(syn[i], syn_type, 0, rand=False)
                synthetic_data.append(Y)
                col_names.append(f'{syn_type}_{i}')
        
        self.num_size += len(synthetic_data)
        self.col_names += col_names
        
        # Combine synthetic columns with original data
        synthetic_data = np.column_stack(synthetic_data)
        combined_data = np.concatenate((synthetic_data, self.data), axis=1)
        
        # Reorganizing categorical column indices
        combined_data_df = pd.DataFrame(combined_data, columns=self.col_names)
        for col in self.col_names[self.num_size:]:
            combined_data_df[col] = pd.Categorical(combined_data_df[col])
            combined_data_df[col] = combined_data_df[col].cat.codes  # This encodes categories from 0 to n-1

        # Convert the DataFrame back to a NumPy array
        combined_data = combined_data_df.to_numpy()
        
        self.quantile_scaler = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(combined_data.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=0,
        )

        # Apply Quantile Transformer to all columns before self.num_size index
        combined_data[:, :self.num_size] = self.quantile_scaler.fit_transform(combined_data[:, :self.num_size])
        
        # Splitting dataset for training and testing
        X_train, X_test = train_test_split(combined_data, test_size=0.1, random_state=42)
        
        # Directory setup for saving CSV files
        save_path = f"synthetic/{self.dataname}"
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save training data
        print("Generating real.csv")
        train = pd.DataFrame(X_train, columns=self.col_names)
        train_path = os.path.join(save_path, "real.csv")
        # save_if_updated(train, train_path)
        train.to_csv(train_path, index=False)
        
        # Save testing data
        print("Generating test.csv")
        test = pd.DataFrame(X_test, columns=self.col_names)
        test_path = os.path.join(save_path, "test.csv")
        # save_if_updated(test, test_path)
        test.to_csv(test_path, index=False)
        
        # Save combined data
        print("Generating syn1.csv")
        combined_path = os.path.join(f"data/{self.dataname}", "syn1.csv")
        # save_if_updated(train, combined_path)
        train.to_csv(combined_path, index=False)
        
        print("Data generation complete!, shape of data: ", X_train.shape)
        
        return X_train
    
    def gen_batch(self, batch_size):
        data = self.gen_data
        if self.current_index >= len(data):
            # Optionally reshuffle here if epochs are handled within this class
            self.shuffled_indices = np.random.permutation(len(data))
            self.current_index = 0

        end_index = min(self.current_index + batch_size, len(data))
        batch_indices = self.shuffled_indices[self.current_index:end_index]
        self.current_index = end_index
        return data[batch_indices]

    def get_category_sizes(self):
        dataT = self.data_cat.T
        return [len(set(x)) for x in dataT]
    
    def get_numerical_sizes(self):
        return self.num_size
    

if __name__ == '__main__':
    # Specify the desired synthetic data type when creating the instance
    np.random.seed(0)
    dataname = 'syn1'
    synth_dataset = SynthDataset(dataname)
    
