import pandas as pd
import numpy as np
from tkinter import *
from tkinter import ttk, filedialog, messagebox, font
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import os
import inspect

class MLAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Assistant")
        self.root.geometry("1100x750")
        self.data = None
        self.original_file_path = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.target_var = StringVar()
        self.problem_type = StringVar(value="classification")
        self.current_font_size = 10
        self.font_family = "Segoe UI"
        self.pipeline_history = []
        self.target_name = None  # Store target name for visualization
        self.create_widgets()
        self.create_menu()
    
    def create_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open CSV", command=self.load_csv)
        file_menu.add_command(label="Save Processed Data", command=self.save_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        view_menu = Menu(menubar, tearoff=0)
        view_menu.add_command(label="Increase Font Size", command=lambda: self.adjust_font_size(1))
        view_menu.add_command(label="Decrease Font Size", command=lambda: self.adjust_font_size(-1))
        view_menu.add_command(label="Reset Font Size", command=lambda: self.adjust_font_size(0))
        menubar.add_cascade(label="View", menu=view_menu)
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
    
    def adjust_font_size(self, change):
        if change == 0:
            self.current_font_size = 10
        else:
            self.current_font_size += change
            if self.current_font_size < 8:
                self.current_font_size = 8
            elif self.current_font_size > 20:
                self.current_font_size = 20
        self.update_fonts()
    
    def update_fonts(self):
        new_font = font.Font(family=self.font_family, size=self.current_font_size)
        self.info_text.configure(font=new_font)
        self.param_text.configure(font=new_font)
        self.eval_text.configure(font=new_font)
        for widget in self.root.winfo_children():
            if isinstance(widget, (Label, Button, ttk.Combobox)):
                try:
                    widget.configure(font=new_font)
                except:
                    pass
        for window in self.root.winfo_children():
            if isinstance(window, Toplevel) and hasattr(window, 'tree'):
                window.tree.configure(font=new_font)
    
    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True)
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_preprocess = ttk.Frame(self.notebook)
        self.tab_model = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="Data")
        self.notebook.add(self.tab_preprocess, text="Preprocessing")
        self.notebook.add(self.tab_model, text="Model")
        self.notebook.add(self.tab_results, text="Results")
        self.build_data_tab()
        self.build_preprocess_tab()
        self.build_model_tab()
        self.build_results_tab()
    
    def build_data_tab(self):
        load_frame = LabelFrame(self.tab_data, text="Load Data", padx=10, pady=10)
        load_frame.pack(fill=X, padx=10, pady=5)
        Button(load_frame, text="Browse CSV File", command=self.load_csv).pack(side=LEFT, padx=5)
        Button(load_frame, text="View Data", command=self.view_data).pack(side=LEFT, padx=5)
        Button(load_frame, text="Create Sample Data", command=self.create_sample_data).pack(side=LEFT, padx=5)
        info_frame = LabelFrame(self.tab_data, text="Data Information", padx=10, pady=10)
        info_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.info_text = Text(info_frame, height=15, wrap=NONE, font=(self.font_family, self.current_font_size))
        scroll_y = Scrollbar(info_frame, orient=VERTICAL, command=self.info_text.yview)
        scroll_x = Scrollbar(info_frame, orient=HORIZONTAL, command=self.info_text.xview)
        self.info_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.pack(side=BOTTOM, fill=X)
        self.info_text.pack(fill=BOTH, expand=True)
        self.info_text.insert(END, "Welcome to Machine Learning Assistant!\n\n")
        self.info_text.insert(END, "To get started:\n")
        self.info_text.insert(END, "1. Load a CSV file using 'Browse CSV File'\n")
        self.info_text.insert(END, "2. Or create sample data for testing using 'Create Sample Data'\n")
        self.info_text.insert(END, "3. Then proceed to Preprocessing tab\n\n")
        self.info_text.insert(END, "Sample data includes missing values for demonstration purposes.")
    
    def build_preprocess_tab(self):
        missing_frame = LabelFrame(self.tab_preprocess, text="Handle Missing Values", padx=10, pady=10)
        missing_frame.pack(fill=X, padx=10, pady=5)
        Label(missing_frame, text="Column:").grid(row=0, column=0, padx=5)
        self.missing_col = ttk.Combobox(missing_frame, state="readonly")
        self.missing_col.grid(row=0, column=1, padx=5)
        Label(missing_frame, text="Action:").grid(row=0, column=2, padx=5)
        self.missing_action = ttk.Combobox(missing_frame, values=[
            "Fill with mean", "Fill with median", "Fill with mode", 
            "Fill with value", "Drop rows", "Drop column"
        ], state="readonly")
        self.missing_action.grid(row=0, column=3, padx=5)
        self.missing_action.bind("<<ComboboxSelected>>", self.show_hide_fill_value)
        self.fill_value_entry = Entry(missing_frame, width=10)
        self.fill_value_entry.grid(row=0, column=4, padx=5)
        self.fill_value_entry.grid_remove()
        Button(missing_frame, text="Apply", command=self.handle_missing_values).grid(row=0, column=5, padx=5)
        convert_frame = LabelFrame(self.tab_preprocess, text="Convert String to Numeric", padx=10, pady=10)
        convert_frame.pack(fill=X, padx=10, pady=5)
        Label(convert_frame, text="Column:").grid(row=0, column=0, padx=5)
        self.convert_col = ttk.Combobox(convert_frame, state="readonly")
        self.convert_col.grid(row=0, column=1, padx=5)
        Label(convert_frame, text="True Value:").grid(row=0, column=2, padx=5)
        self.true_value = Entry(convert_frame, width=10)
        self.true_value.insert(0, "true")
        self.true_value.grid(row=0, column=3, padx=5)
        Label(convert_frame, text="False Value:").grid(row=0, column=4, padx=5)
        self.false_value = Entry(convert_frame, width=10)
        self.false_value.insert(0, "false")
        self.false_value.grid(row=0, column=5, padx=5)
        Button(convert_frame, text="Convert", command=self.convert_string_to_numeric).grid(row=0, column=6, padx=5)
        encode_frame = LabelFrame(self.tab_preprocess, text="Encode Categorical Data", padx=10, pady=10)
        encode_frame.pack(fill=X, padx=10, pady=5)
        Label(encode_frame, text="Method:").pack(side=LEFT, padx=5)
        self.encode_method = ttk.Combobox(encode_frame, values=["Label Encoding", "One-Hot Encoding"], state="readonly")
        self.encode_method.pack(side=LEFT, padx=5)
        Button(encode_frame, text="Apply Encoding", command=self.encode_categorical).pack(side=LEFT, padx=5)
        split_frame = LabelFrame(self.tab_preprocess, text="Split Data", padx=10, pady=10)
        split_frame.pack(fill=X, padx=10, pady=5)
        Label(split_frame, text="Target Variable:").grid(row=0, column=0, padx=5)
        self.target_combobox = ttk.Combobox(split_frame, textvariable=self.target_var, state="readonly")
        self.target_combobox.grid(row=0, column=1, padx=5)
        Label(split_frame, text="Test Size:").grid(row=0, column=2, padx=5)
        self.test_size = Entry(split_frame, width=5)
        self.test_size.insert(0, "0.2")
        self.test_size.grid(row=0, column=3, padx=5)
        Label(split_frame, text="Random State:").grid(row=0, column=4, padx=5)
        self.random_state = Entry(split_frame, width=5)
        self.random_state.insert(0, "42")
        self.random_state.grid(row=0, column=5, padx=5)
        Button(split_frame, text="Split Data", command=self.split_data).grid(row=0, column=6, padx=5)
        scale_frame = LabelFrame(self.tab_preprocess, text="Scale Data", padx=10, pady=10)
        scale_frame.pack(fill=X, padx=10, pady=5)
        self.scale_method = ttk.Combobox(scale_frame, values=["Standard Scaler", "MinMax Scaler"], state="readonly")
        self.scale_method.pack(side=LEFT, padx=5)
        Button(scale_frame, text="Apply Scaling", command=self.scale_data).pack(side=LEFT, padx=5)
    
    def build_model_tab(self):
        model_frame = LabelFrame(self.tab_model, text="Model Selection", padx=10, pady=10)
        model_frame.pack(fill=X, padx=10, pady=5)
        Label(model_frame, text="Problem Type:").grid(row=0, column=0, padx=5)
        self.problem_combobox = ttk.Combobox(model_frame, textvariable=self.problem_type, 
                                          values=["classification", "regression"], state="readonly")
        self.problem_combobox.grid(row=0, column=1, padx=5)
        Label(model_frame, text="Model:").grid(row=0, column=2, padx=5)
        self.model_combobox = ttk.Combobox(model_frame, state="readonly")
        self.model_combobox.grid(row=0, column=3, padx=5)
        self.problem_combobox.bind("<<ComboboxSelected>>", self.update_model_options)
        Button(model_frame, text="Train Model", command=self.train_model).grid(row=0, column=4, padx=5)
        param_frame = LabelFrame(self.tab_model, text="Model Parameters", padx=10, pady=10)
        param_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.param_text = Text(param_frame, height=10, font=(self.font_family, self.current_font_size))
        self.param_text.pack(fill=BOTH, expand=True)
        self.param_text.insert(END, "n_estimators=100\nrandom_state=42\nmax_depth=None")
    
    def build_results_tab(self):
        eval_frame = LabelFrame(self.tab_results, text="Model Evaluation", padx=10, pady=10)
        eval_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.eval_text = Text(eval_frame, height=15, font=(self.font_family, self.current_font_size))
        scroll = Scrollbar(eval_frame, command=self.eval_text.yview)
        self.eval_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=RIGHT, fill=Y)
        self.eval_text.pack(fill=BOTH, expand=True)
        vis_frame = LabelFrame(self.tab_results, text="Visualization", padx=10, pady=10)
        vis_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.figure = plt.figure(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        btn_frame = Frame(vis_frame)
        btn_frame.pack(fill=X, pady=5)
        Button(btn_frame, text="Plot Feature Importance", command=self.plot_feature_importance).pack(side=LEFT, padx=5)
        Button(btn_frame, text="Plot Confusion Matrix", command=self.plot_confusion_matrix).pack(side=LEFT, padx=5)
        # NEW: Add 2D Visualization button
        Button(btn_frame, text="2D Visualization", command=self.show_2d_visualization).pack(side=LEFT, padx=5)
        Button(btn_frame, text="Clear Plot", command=self.clear_plot).pack(side=LEFT, padx=5)
        Button(btn_frame, text="Generate Pipeline Code", command=self.generate_pipeline_code).pack(side=LEFT, padx=5)
    
    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            try:
                self.original_file_path = filepath
                self.data = pd.read_csv(filepath)
                self.update_data_info()
                self.update_column_comboboxes()
                self.track_action("load_data", {"file_path": filepath})
                messagebox.showinfo("Success", f"Data loaded successfully!\nShape: {self.data.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def create_sample_data(self):
        data = {
            'Age': [25, 30, 35, np.nan, 45, 28, 32, 40, np.nan, 50],
            'Income': [50000, 60000, np.nan, 80000, 90000, 55000, 65000, np.nan, 75000, 95000],
            'Education': ['High School', 'College', 'College', 'PhD', 'Masters', 
                         'High School', 'College', 'Masters', np.nan, 'PhD'],
            'Purchased': ['No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
        }
        self.data = pd.DataFrame(data)
        self.update_data_info()
        self.update_column_comboboxes()
        messagebox.showinfo("Sample Data", "Sample dataset created with missing values for demonstration!")
    
    def view_data(self):
        if self.data is not None:
            data_window = Toplevel(self.root)
            data_window.title("Data Viewer")
            tree = ttk.Treeview(data_window)
            tree["columns"] = list(self.data.columns)
            tree["show"] = "headings"
            tree.column("#0", width=0, stretch=NO)
            for col in self.data.columns:
                tree.column(col, anchor=W, width=100)
                tree.heading(col, text=col, anchor=W)
            for i, row in self.data.head(50).iterrows():
                tree.insert("", END, values=list(row))
            scroll_y = ttk.Scrollbar(data_window, orient=VERTICAL, command=tree.yview)
            scroll_x = ttk.Scrollbar(data_window, orient=HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
            tree.pack(side=LEFT, fill=BOTH, expand=True)
            scroll_y.pack(side=RIGHT, fill=Y)
            scroll_x.pack(side=BOTTOM, fill=X)
            data_window.tree = tree
        else:
            messagebox.showwarning("Warning", "No data loaded!")
    
    def update_data_info(self):
        if self.data is not None:
            info = f"Shape: {self.data.shape}\n\n"
            info += "Columns:\n" + "\n".join(self.data.columns) + "\n\n"
            info += "Data Types:\n" + str(self.data.dtypes) + "\n\n"
            info += "Missing Values:\n" + str(self.data.isnull().sum())
            self.info_text.delete(1.0, END)
            self.info_text.insert(END, info)
    
    def update_column_comboboxes(self):
        if self.data is not None:
            columns = list(self.data.columns)
            self.missing_col['values'] = columns
            self.target_combobox['values'] = columns
            self.convert_col['values'] = columns
            if len(columns) > 0:
                self.missing_col.current(0)
                self.target_combobox.current(0)
                self.convert_col.current(0)
    
    def show_hide_fill_value(self, event=None):
        if self.missing_action.get() == "Fill with value":
            self.fill_value_entry.grid()
        else:
            self.fill_value_entry.grid_remove()
    
    def handle_missing_values(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded!")
            return
        col = self.missing_col.get()
        action = self.missing_action.get()
        if not col or not action:
            messagebox.showwarning("Warning", "Please select column and action")
            return
        try:
            if action == "Fill with mean":
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            elif action == "Fill with median":
                self.data[col].fillna(self.data[col].median(), inplace=True)
            elif action == "Fill with mode":
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            elif action == "Fill with value":
                value = self.fill_value_entry.get()
                if self.data[col].dtype.kind in 'iufc':
                    try:
                        value = float(value) if '.' in value else int(value)
                    except ValueError:
                        pass
                self.data[col].fillna(value, inplace=True)
            elif action == "Drop rows":
                self.data.dropna(subset=[col], inplace=True)
            elif action == "Drop column":
                self.data.drop(columns=[col], inplace=True)
            self.update_data_info()
            self.update_column_comboboxes()
            self.track_action("handle_missing_values", {
                "col": col,
                "action": action,
                "value": value if action == "Fill with value" else None
            })
            messagebox.showinfo("Success", "Missing values handled successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to handle missing values:\n{str(e)}")
    
    def convert_string_to_numeric(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded!")
            return
        col = self.convert_col.get()
        true_val = self.true_value.get().strip()
        false_val = self.false_value.get().strip()
        if not col:
            messagebox.showwarning("Warning", "Please select a column")
            return
        try:
            if self.data[col].dtype == 'object':
                if true_val and false_val:
                    mapping = {true_val: 1, false_val: 0}
                    self.data[col] = self.data[col].map(mapping).fillna(self.data[col])
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    if self.data[col].dtype == 'object':
                        messagebox.showinfo("Info", "Column contains non-numeric values that couldn't be converted")
                    else:
                        messagebox.showinfo("Success", "String values converted to numeric")
                except Exception as e:
                    messagebox.showerror("Error", f"Conversion failed: {str(e)}")
            else:
                messagebox.showinfo("Info", "Column is not string type")
            self.update_data_info()
            self.track_action("convert_string_to_numeric", {
                "col": col,
                "true_val": true_val,
                "false_val": false_val
            })
        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert values: {str(e)}")
    
    def encode_categorical(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded!")
            return
        method = self.encode_method.get()
        if not method:
            messagebox.showwarning("Warning", "Please select encoding method")
            return
        try:
            cat_cols = self.data.select_dtypes(exclude=[np.number]).columns
            if len(cat_cols) == 0:
                messagebox.showinfo("Info", "No categorical columns found!")
                return
            if method == "Label Encoding":
                for col in cat_cols:
                    self.data[col] = LabelEncoder().fit_transform(self.data[col])
            elif method == "One-Hot Encoding":
                self.data = pd.get_dummies(self.data, columns=cat_cols, drop_first=True)
            self.update_data_info()
            self.update_column_comboboxes()
            self.track_action("encode_categorical", {"method": method})
            messagebox.showinfo("Success", "Encoding applied successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to encode data:\n{str(e)}")
    
    def split_data(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded!")
            return
        target = self.target_var.get()
        if not target:
            messagebox.showwarning("Warning", "Please select target variable")
            return
        try:
            # Store target name for visualization
            self.target_name = target
            
            test_size = float(self.test_size.get())
            random_state = int(self.random_state.get())
            X = self.data.drop(columns=[target])
            y = self.data[target]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            if y.dtype.kind in 'iufc':
                unique_vals = len(np.unique(y))
                if unique_vals / len(y) < 0.05 or unique_vals < 10:
                    self.problem_type.set("classification")
                else:
                    self.problem_type.set("regression")
            else:
                self.problem_type.set("classification")
            self.update_model_options()
            self.track_action("split_data", {
                "target": target,
                "test_size": test_size,
                "random_state": random_state
            })
            messagebox.showinfo("Success", 
                f"Data split successfully!\nTrain: {self.X_train.shape[0]} samples\nTest: {self.X_test.shape[0]} samples")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to split data:\n{str(e)}")
    
    def scale_data(self):
        if self.X_train is None:
            messagebox.showwarning("Warning", "Please split data first!")
            return
        method = self.scale_method.get()
        if not method:
            messagebox.showwarning("Warning", "Please select scaling method")
            return
        try:
            if method == "Standard Scaler":
                scaler = StandardScaler()
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
            elif method == "MinMax Scaler":
                scaler = MinMaxScaler()
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
            self.track_action("scale_data", {"method": method})
            messagebox.showinfo("Success", "Data scaled successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scale data:\n{str(e)}")
    
    def update_model_options(self, event=None):
        if self.problem_type.get() == "classification":
            self.model_combobox['values'] = [
                "Logistic Regression", 
                "Random Forest Classifier", 
                "SVC", 
                "Gradient Boosting Classifier"
            ]
        else:
            self.model_combobox['values'] = [
                "Linear Regression", 
                "Random Forest Regressor", 
                "SVR", 
                "Gradient Boosting Regressor"
            ]
        if len(self.model_combobox['values']) > 0:
            self.model_combobox.current(0)
    
    def train_model(self):
        if self.X_train is None:
            messagebox.showwarning("Warning", "Please split data first!")
            return
        model_name = self.model_combobox.get()
        if not model_name:
            messagebox.showwarning("Warning", "Please select a model")
            return
        try:
            model_mapping = {
                "Logistic Regression": LogisticRegression,
                "Random Forest Classifier": RandomForestClassifier,
                "SVC": SVC,
                "Gradient Boosting Classifier": GradientBoostingClassifier,
                "Linear Regression": LinearRegression,
                "Random Forest Regressor": RandomForestRegressor,
                "SVR": SVR,
                "Gradient Boosting Regressor": GradientBoostingRegressor
            }
            allowed_params = {
                "Logistic Regression": LogisticRegression().get_params().keys(),
                "Random Forest Classifier": RandomForestClassifier().get_params().keys(),
                "SVC": SVC().get_params().keys(),
                "Gradient Boosting Classifier": GradientBoostingClassifier().get_params().keys(),
                "Linear Regression": LinearRegression().get_params().keys(),
                "Random Forest Regressor": RandomForestRegressor().get_params().keys(),
                "SVR": SVR().get_params().keys(),
                "Gradient Boosting Regressor": GradientBoostingRegressor().get_params().keys()
            }
            raw_params = self.param_text.get("1.0", END).split('\n')
            params = {}
            for line in raw_params:
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        if value.lower() == 'none':
                            value = None
                        elif value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        else:
                            pass
                    if key in allowed_params[model_name]:
                        params[key] = value
            model_class = model_mapping[model_name]
            self.model = model_class(**params)
            self.model.fit(self.X_train, self.y_train)
            self.evaluate_model()
            self.track_action("train_model", {
                "model_name": model_name,
                "params": params
            })
            messagebox.showinfo("Success", f"{model_name} trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model:\n{str(e)}")
    
    def track_action(self, action_name, params):
        self.pipeline_history.append({
            "action": action_name,
            "params": params,
            "timestamp": pd.Timestamp.now()
        })
    
    def generate_pipeline_code(self):
        if not self.pipeline_history:
            messagebox.showinfo("Info", "No workflow steps recorded yet")
            return
        try:
            code = "# Machine Learning Pipeline Generated by ML Assistant\n"
            code += "import pandas as pd\nimport numpy as np\n"
            code += "from sklearn.model_selection import train_test_split\n"
            code += "from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder\n"
            code += "from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score\n"
            code += "import matplotlib.pyplot as plt\n\n"
            if self.original_file_path:
                code += f'data = pd.read_csv("{self.original_file_path}")\n'
                code += f'print("Data loaded. Shape:", data.shape)\n\n'
            for step in self.pipeline_history:
                action = step["action"]
                params = step["params"]
                if action == "handle_missing_values":
                    col = params["col"]
                    action_type = params["action"]
                    if action_type == "Fill with mean":
                        code += f"data['{col}'].fillna(data['{col}'].mean(), inplace=True)\n"
                    elif action_type == "Fill with median":
                        code += f"data['{col}'].fillna(data['{col}'].median(), inplace=True)\n"
                    elif action_type == "Fill with mode":
                        code += f"data['{col}'].fillna(data['{col}'].mode()[0], inplace=True)\n"
                    elif action_type == "Fill with value":
                        value = params["value"]
                        code += f"data['{col}'].fillna({value}, inplace=True)\n"
                    elif action_type == "Drop rows":
                        code += f"data.dropna(subset=['{col}'], inplace=True)\n"
                    elif action_type == "Drop column":
                        code += f"data.drop(columns=['{col}'], inplace=True)\n"
                    code += "\n"
                elif action == "convert_string_to_numeric":
                    col = params["col"]
                    true_val = params["true_val"]
                    false_val = params["false_val"]
                    code += f"mapping = {{'{true_val}': 1, '{false_val}': 0}}\n"
                    code += f"data['{col}'] = data['{col}'].map(mapping).fillna(data['{col}'])\n"
                    code += f"data['{col}'] = pd.to_numeric(data['{col}'], errors='coerce')\n\n"
                elif action == "encode_categorical":
                    method = params["method"]
                    if method == "Label Encoding":
                        code += "le = LabelEncoder()\n"
                        code += "for col in data.select_dtypes(exclude=[np.number]).columns:\n"
                        code += "    data[col] = le.fit_transform(data[col])\n"
                    elif method == "One-Hot Encoding":
                        code += "data = pd.get_dummies(data, columns=data.select_dtypes(exclude=[np.number]).columns, drop_first=True)\n"
                    code += "\n"
                elif action == "split_data":
                    target = params["target"]
                    test_size = params["test_size"]
                    random_state = params["random_state"]
                    code += f"X = data.drop(columns=['{target}'])\n"
                    code += f"y = data['{target}']\n"
                    code += f"X_train, X_test, y_train, y_test = train_test_split(\n"
                    code += f"    X, y, test_size={test_size}, random_state={random_state}\n"
                    code += ")\n\n"
                elif action == "scale_data":
                    method = params["method"]
                    if method == "Standard Scaler":
                        code += "scaler = StandardScaler()\n"
                    elif method == "MinMax Scaler":
                        code += "scaler = MinMaxScaler()\n"
                    code += "X_train = scaler.fit_transform(X_train)\n"
                    code += "X_test = scaler.transform(X_test)\n\n"
                elif action == "train_model":
                    model_name = params["model_name"]
                    model_params = params["params"]
                    model_class = type(self.model).__name__
                    module = inspect.getmodule(self.model).__name__
                    param_str = ",\n    ".join([f"{k}={repr(v)}" for k, v in model_params.items()])
                    code += f"from {module} import {model_class}\n"
                    code += f"model = {model_class}(\n    {param_str}\n)\n"
                    code += "model.fit(X_train, y_train)\n\n"
                    code += "train_pred = model.predict(X_train)\n"
                    code += "test_pred = model.predict(X_test)\n\n"
                    code += "if hasattr(model, 'predict_proba'):\n"
                    code += "    print('Training Accuracy:', accuracy_score(y_train, train_pred))\n"
                    code += "    print('Training F1 Score:', f1_score(y_train, train_pred, average='weighted'))\n"
                    code += "    print('Test Accuracy:', accuracy_score(y_test, test_pred))\n"
                    code += "    print('Test F1 Score:', f1_score(y_test, test_pred, average='weighted'))\n"
                    code += "else:\n"
                    code += "    print('Training MSE:', mean_squared_error(y_train, train_pred))\n"
                    code += "    print('Training R²:', r2_score(y_train, train_pred))\n"
                    code += "    print('Test MSE:', mean_squared_error(y_test, test_pred))\n"
                    code += "    print('Test R²:', r2_score(y_test, test_pred))\n\n"
            code += "if hasattr(model, 'feature_importances_'):\n"
            code += "    plt.figure(figsize=(10, 6))\n"
            code += "    importances = model.feature_importances_\n"
            code += "    features = X.columns if hasattr(X, 'columns') else range(len(importances))\n"
            code += "    sorted_idx = np.argsort(importances)\n"
            code += "    plt.barh(range(len(importances)), importances[sorted_idx], align='center')\n"
            code += "    plt.yticks(range(len(importances)), [features[i] for i in sorted_idx])\n"
            code += "    plt.xlabel('Feature Importance')\n"
            code += "    plt.title('Feature Importance')\n"
            code += "    plt.tight_layout()\n"
            code += "    plt.show()\n\n"
            code += "if hasattr(model, 'predict_proba'):\n"
            code += "    from sklearn.metrics import confusion_matrix\n"
            code += "    cm = confusion_matrix(y_test, test_pred)\n"
            code += "    plt.figure(figsize=(8, 6))\n"
            code += "    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)\n"
            code += "    plt.title('Confusion Matrix')\n"
            code += "    plt.colorbar()\n"
            code += "    plt.xlabel('Predicted Label')\n"
            code += "    plt.ylabel('True Label')\n"
            code += "    plt.show()\n"
            file_path = filedialog.asksaveasfilename(
                defaultextension=".py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, "w") as f:
                    f.write(code)
                messagebox.showinfo("Success", f"Pipeline code saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate code:\n{str(e)}")
    
    def evaluate_model(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No model trained yet!")
            return
        try:
            self.eval_text.delete(1.0, END)
            train_pred = self.model.predict(self.X_train)
            self.eval_text.insert(END, "=== Training Set Performance ===\n")
            if hasattr(self.model, 'predict_proba'):
                train_acc = accuracy_score(self.y_train, train_pred)
                train_f1 = f1_score(self.y_train, train_pred, average='weighted')
                self.eval_text.insert(END, f"Accuracy: {train_acc:.4f}\n")
                self.eval_text.insert(END, f"F1 Score: {train_f1:.4f}\n")
            else:
                train_mse = mean_squared_error(self.y_train, train_pred)
                train_r2 = r2_score(self.y_train, train_pred)
                self.eval_text.insert(END, f"MSE: {train_mse:.4f}\n")
                self.eval_text.insert(END, f"R² Score: {train_r2:.4f}\n")
            test_pred = self.model.predict(self.X_test)
            self.eval_text.insert(END, "\n=== Test Set Performance ===\n")
            if hasattr(self.model, 'predict_proba'):
                test_acc = accuracy_score(self.y_test, test_pred)
                test_f1 = f1_score(self.y_test, test_pred, average='weighted')
                self.eval_text.insert(END, f"Accuracy: {test_acc:.4f}\n")
                self.eval_text.insert(END, f"F1 Score: {test_f1:.4f}\n")
            else:
                test_mse = mean_squared_error(self.y_test, test_pred)
                test_r2 = r2_score(self.y_test, test_pred)
                self.eval_text.insert(END, f"MSE: {test_mse:.4f}\n")
                self.eval_text.insert(END, f"R² Score: {test_r2:.4f}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to evaluate model:\n{str(e)}")
    
    def plot_feature_importance(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No model trained yet!")
            return
        try:
            if hasattr(self.model, 'feature_importances_'):
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                importances = self.model.feature_importances_
                if hasattr(self.X_train, 'columns'):
                    features = self.X_train.columns
                elif hasattr(self.X_train, 'name'):
                    features = [f"Feature {i}" for i in range(len(importances))]
                else:
                    features = range(len(importances))
                sorted_idx = np.argsort(importances)
                sorted_features = [features[i] for i in sorted_idx]
                sorted_importances = importances[sorted_idx]
                ax.barh(sorted_features, sorted_importances)
                ax.set_title("Feature Importance")
                ax.set_xlabel("Importance")
                self.canvas.draw()
            else:
                messagebox.showinfo("Info", "This model doesn't support feature importance")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot feature importance:\n{str(e)}")
    
    def plot_confusion_matrix(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No model trained yet!")
            return
        try:
            if hasattr(self.model, 'predict_proba'):
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                y_pred = self.model.predict(self.X_test)
                cm = confusion_matrix(self.y_test, y_pred)
                cax = ax.matshow(cm, cmap=plt.cm.Blues)
                self.figure.colorbar(cax)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_xticks(range(cm.shape[1]))
                ax.set_yticks(range(cm.shape[0]))
                if hasattr(self.y_test, 'name'):
                    class_names = self.y_test.unique()
                    ax.set_xticklabels(class_names)
                    ax.set_yticklabels(class_names)
                self.canvas.draw()
            else:
                messagebox.showinfo("Info", "Confusion matrix is only for classification")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot confusion matrix:\n{str(e)}")
    
    # NEW: 2D Visualization feature
    def show_2d_visualization(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data loaded!")
            return
        if not hasattr(self, 'X_train') or self.X_train is None:
            messagebox.showwarning("Warning", "Please split data first!")
            return
        
        # Create selection dialog
        dialog = Toplevel(self.root)
        dialog.title("2D Visualization Options")
        dialog.geometry("400x300")
        dialog.grab_set()
        
        # Feature selection
        features = self.data.columns.tolist()
        if self.target_name in features:
            features.remove(self.target_name)
            
        Label(dialog, text="X-axis Feature:").pack(pady=(10, 0))
        x_var = StringVar()
        x_combobox = ttk.Combobox(dialog, textvariable=x_var, values=features, state="readonly")
        x_combobox.pack()
        if features: x_combobox.current(0)
        
        Label(dialog, text="Y-axis Feature:").pack(pady=(10, 0))
        y_var = StringVar()
        y_combobox = ttk.Combobox(dialog, textvariable=y_var, values=features, state="readonly")
        y_combobox.pack()
        if len(features) > 1: y_combobox.current(1)
        
        # Coloring option
        color_var = BooleanVar(value=True)
        Checkbutton(dialog, text="Color by target variable", variable=color_var).pack(pady=(10, 0))
        
        # Show train/test split
        split_var = BooleanVar(value=True)
        Checkbutton(dialog, text="Show train/test split", variable=split_var).pack(pady=(5, 0))
        
        # Apply button
        Button(dialog, text="Generate Plot", 
              command=lambda: self.plot_2d_visualization(
                  x_var.get(), y_var.get(), 
                  color_var.get(), split_var.get(), dialog)
              ).pack(pady=15)
    
    def plot_2d_visualization(self, x_feature, y_feature, color_by_target, show_split, dialog):
        if not x_feature or not y_feature:
            messagebox.showwarning("Warning", "Please select both X and Y features")
            return
            
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Get data for plotting
            x_data = self.data[x_feature]
            y_data = self.data[y_feature]
            
            # Get indices for train/test split if needed
            if show_split:
                train_idx = self.X_train.index
                test_idx = self.X_test.index
            else:
                train_idx = test_idx = self.data.index
                
            # Plot training data
            if color_by_target and self.target_name:
                # Color by target variable
                if self.problem_type.get() == "classification":
                    # Classification - discrete colors
                    unique_targets = self.data[self.target_name].unique()
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_targets)))
                    color_map = {target: color for target, color in zip(unique_targets, colors)}
                    
                    # Plot each class separately
                    for target in unique_targets:
                        target_idx = self.data[self.data[self.target_name] == target].index
                        train_target_idx = target_idx.intersection(train_idx)
                        test_target_idx = target_idx.intersection(test_idx)
                        
                        # Plot training points
                        ax.scatter(
                            x_data[train_target_idx], y_data[train_target_idx],
                            color=color_map[target], marker='o', alpha=0.7,
                            edgecolors='k', s=60, label=f"{target} (Train)"
                        )
                        
                        # Plot test points
                        if show_split and test_target_idx.any():
                            ax.scatter(
                                x_data[test_target_idx], y_data[test_target_idx],
                                color=color_map[target], marker='s', alpha=0.7,
                                edgecolors='k', s=60, label=f"{target} (Test)"
                            )
                else:
                    # Regression - continuous color scale
                    target_data = self.data[self.target_name]
                    train_target = target_data.loc[train_idx]
                    test_target = target_data.loc[test_idx]
                    
                    # Plot training points
                    sc_train = ax.scatter(
                        x_data[train_idx], y_data[train_idx],
                        c=train_target, cmap='viridis', marker='o', alpha=0.7,
                        edgecolors='k', s=60, label="Train"
                    )
                    
                    # Plot test points
                    if show_split:
                        sc_test = ax.scatter(
                            x_data[test_idx], y_data[test_idx],
                            c=test_target, cmap='viridis', marker='s', alpha=0.7,
                            edgecolors='k', s=60, label="Test"
                        )
                    
                    # Add colorbar
                    plt.colorbar(sc_train, ax=ax, label=self.target_name)
            else:
                # Single color
                # Plot training points
                ax.scatter(
                    x_data[train_idx], y_data[train_idx],
                    color='blue', marker='o', alpha=0.7,
                    edgecolors='k', s=60, label="Train"
                )
                
                # Plot test points
                if show_split:
                    ax.scatter(
                        x_data[test_idx], y_data[test_idx],
                        color='red', marker='s', alpha=0.7,
                        edgecolors='k', s=60, label="Test"
                    )
            
            # Add decision boundary if model exists and is 2D classifier
            if (self.model is not None and 
                hasattr(self.model, 'predict') and 
                self.problem_type.get() == "classification"):
                self.plot_decision_boundary(ax, x_feature, y_feature)
            
            # Add labels and title
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            title = f"{y_feature} vs {x_feature}"
            if color_by_target and self.target_name:
                title += f" (Colored by {self.target_name})"
            ax.set_title(title)
            
            # Add legend
            ax.legend(loc='best')
            
            # Draw the plot
            self.canvas.draw()
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create 2D plot:\n{str(e)}")
    
    def plot_decision_boundary(self, ax, x_feature, y_feature):
        """Plot decision boundary for classification models"""
        try:
            # Get the current model
            model = self.model
            
            # Create a grid to plot decision boundary
            x_min, x_max = self.data[x_feature].min() - 1, self.data[x_feature].max() + 1
            y_min, y_max = self.data[y_feature].min() - 1, self.data[y_feature].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
            
            # Prepare grid points for prediction
            grid = np.c_[xx.ravel(), yy.ravel()]
            
            # Get feature indices
            features = self.X_train.columns.tolist()
            x_idx = features.index(x_feature)
            y_idx = features.index(y_feature)
            
            # Create a full feature matrix with zeros
            grid_full = np.zeros((grid.shape[0], len(features)))
            
            # Set the selected features
            grid_full[:, x_idx] = grid[:, 0]
            grid_full[:, y_idx] = grid[:, 1]
            
            # Predict on grid
            Z = model.predict(grid_full)
            
            # Reshape and plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)
        except:
            # Silently fail if we can't plot decision boundary
            pass
    
    def clear_plot(self):
        self.figure.clear()
        self.canvas.draw()
    
    def save_data(self):
        if self.data is None:
            messagebox.showwarning("Warning", "No data to save!")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.data.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Data saved successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data:\n{str(e)}")
    
    def show_about(self):
        about_text = "Machine Learning Assistant\n\n"
        about_text += "Version: 3.0\n"
        about_text += "Developed by: Your Name\n\n"
        about_text += "Features:\n"
        about_text += "- Load and explore CSV data\n"
        about_text += "- Handle missing values\n"
        about_text += "- Convert strings to numeric\n"
        about_text += "- Encode categorical data\n"
        about_text += "- Split data into train/test sets\n"
        about_text += "- Scale features\n"
        about_text += "- Train machine learning models\n"
        about_text += "- Evaluate model performance\n"
        about_text += "- Visualize results\n"
        about_text += "- Generate pipeline code\n\n"
        about_text += "Use View menu to adjust font size"
        messagebox.showinfo("About Machine Learning Assistant", about_text)

if __name__ == "__main__":
    root = Tk()
    app = MLAssistantGUI(root)
    root.mainloop()