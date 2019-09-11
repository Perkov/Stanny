import json
import os
import sys
from itertools import islice
from pathlib import Path

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QSortFilterProxyModel, QCoreApplication, QAbstractTableModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QFont
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QTextEdit, QTableView, QErrorMessage, \
    QCompleter, QComboBox

plt.style.use('_classic_test')

KS_CLIP = f'{os.getcwd()}/stanista_clippana'

def parse_filenames(path):
    all_paths = islice(list(Path(path).rglob("*.shp")), 1000)
    all_paths = [str(i) for i in all_paths]
    file_names = [os.path.split(i)[1] for i in all_paths[1:]]
    file_names = [' '.join(i.split('_')[1:]).split('.')[0] for i in file_names]
    file_names.insert(0, "<unesi naziv podrucja EM>")
    return file_names, all_paths


FILE_NAMES = parse_filenames(KS_CLIP)[0]
ALL_PATHS = parse_filenames(KS_CLIP)[1]
PARENT_DIR = os.getcwd()

with open(f'{PARENT_DIR}/nks/annex_dict_.json', 'r', encoding="utf8") as a, open(
        f'{PARENT_DIR}/nks/povs_dict_2.json', 'r', encoding="utf8"
) as b, open(f'{PARENT_DIR}/nks/data.json', 'r', encoding="utf8") as c:
    DICT_annex = json.load(a)
    DICT_povs = json.load(b)
    DICT_nazivi = json.load(c)


def mask_ciljna_stanista(row):
    if row['broj_ciljnih'] == 2:
        row['ciljni'] = row['ciljni'].split(',')
        row2 = row.copy()
        row['ciljni'] = row['ciljni'][0]
        row2['ciljni'] = row2['ciljni'][1]
        row.replace({row2['ciljni']: ' ***'}, regex=True, inplace=True)
        row2.replace({row['ciljni']: '***'}, regex=True, inplace=True)
        return pd.concat([row, row2], axis=1)
    return row


def f_max(row):
    if len(row['temp']) == 3 and row['ciljni'] in row['temp'][0]:
        val = row['impact_area']*0.65
    elif len(row['temp']) == 3 and row['ciljni'] in row['temp'][1]:
        val = row['impact_area']*0.40
    elif len(row['temp']) == 3 and row['ciljni'] in row['temp'][2]:
        val = row['impact_area']*0.25
    elif len(row['temp']) == 2 and row['ciljni'] in row['temp'][0]:
        val = row['impact_area']*0.85
    elif len(row['temp']) == 2 and row['ciljni'] in row['temp'][1]:
        val = row['impact_area']*0.45
    else:
        val = row['impact_area']
    return val


def f_min(row):
    if len(row['temp']) == 3 and row['ciljni'] in row['temp'][0]:
        val = row['impact_area']*0.34
    elif len(row['temp']) == 3 and row['ciljni'] in row['temp'][1]:
        val = row['impact_area']*0.20
    elif len(row['temp']) == 3 and row['ciljni'] in row['temp'][2]:
        val = row['impact_area']*0.15
    elif len(row['temp']) == 2 and row['ciljni'] in row['temp'][0]:
        val = row['impact_area']*0.46
    elif len(row['temp']) == 2 and row['ciljni'] in row['temp'][1]:
        val = row['impact_area']*0.15
    else:
        val = row['impact_area']*0.85
    return val


def calculate(row):
    if row['b'] == None and row['c'] == None:
        val = row['AREA_HA']
    elif row['c'] == None:
        val = [row['AREA_HA']*0.80, row['AREA_HA']*0.45]
    else:
        val = [
            row['AREA_HA']*0.65,
            row['AREA_HA']*0.40,
            row['AREA_HA']*0.25
        ]
    return val


class FileNames():
    """
    Walks through selected folder and returns list of paths of all files.
    If number of files exceeds 1000, operation stops.
    Path strings are transformed to filenames.
    """
    def __init__(self, path):
        all_paths = islice(list(Path(path).rglob("*.*")), 1000)
        self.file_names = [
            os.path.basename(i) for i in all_paths
        ]


class Window(QMainWindow):
    """
    Contains main window GUI and methods for getting the selected path,
    passing it to Parser object, and initializing graph if there is data.
    """

    def __init__(self):
        super().__init__()
        self.shapefile = None
        self.clipfile_path = None
        self.output_gdf =  None
        self.graph = None
        self.folder_name = None
        self.popup = None
        self.text = None
        self.loader = None
        self.combo = None
        self.combo_selected_text = None
        self.calculator = None
        self.error_dialog = None
        self.impact_gdf = None
        self.impact_ciljni = None
        self.impact_model = None
        self.title = "Stanny v0.1"
        self.top = 100
        self.left = 100
        self.width = 620
        self.height = 440
        self.window_ui()

    def window_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.text = QTextEdit(self)
        self.combo = ExtendedCombo(self)
        self.table = QTableView()
        self.table.move(10, 400)
        self.table.setGeometry(50, 100, 800, 600)
        self.text.setGeometry(0, 400, 300, 190)
        self.combo.setGeometry(0, 80, 300, 50)
        self.combo.move(20, 30)
        self.text.move(20, 100)
        self.buttons()
        self.combo_box()
        self.show()

    def buttons(self):
        button = QPushButton("Potvrdite odabir", self)
        button.setGeometry(200, 200, 130, 31)
        button.move(335, 40)
        button.clicked.connect(self.update)

        button2 = QPushButton('Prikaz POVS-a', self)
        button2.setGeometry(200, 200, 120, 30)
        button2.move(480, 40)
        button2.clicked.connect(self.popup_with_graph)

        button3 = QPushButton('Odaberite File s kojim clipate', self)
        button3.setGeometry(200, 200, 265, 30)
        button3.move(335, 132)
        button3.clicked.connect(self.folder_path)

        button4 = QPushButton('Prikazi clippane povrsine', self)
        button4.setGeometry(200, 200, 265, 30)
        button4.move(335, 300)
        button4.clicked.connect(self.show_table)

        button5 = QPushButton('Tablica relativnog gubitka povrsina', self)
        button5.setGeometry(200, 200, 265, 30)
        button5.move(335, 340)
        button5.clicked.connect(self.show_loss_all)

        button6 = QPushButton('Tablica gubitka natura stanista', self)
        button6.setGeometry(200, 200, 265, 30)
        button6.move(335, 380)
        button6.clicked.connect(self.show_loss_n2k)

    def update(self):
        if '<' not in self.combo.currentText():
            self.text.setText(
                f' \nPdrucje EM koje cu klippati: \n{self.combo.currentText()}\n'
            )
            self.shapefile = GetShapeFile(TextInstance(self.combo.currentText()).text)
            self.graph = ShapefileLoader(self.shapefile.path).gdf
        else:
            self.error_dialog = QErrorMessage()
            self.error_dialog.showMessage('Niste odabrali podrucje s liste!')

    def folder_path(self):
        "Getting path of selected folder and passing to loading method."
        f_path = QFileDialog.getOpenFileName(self, 'odaberi shapefile s kojim clippas')
        if 'shp' in f_path[0]:
            self.load_after_getting_path(f_path)
            self.popup_clipped_shp()
            self.calculate_loss()
        else:
            self.error_dialog = QErrorMessage()
            self.error_dialog.showMessage('Niste odabrali shapefile (.shp)!')

    def combo_box(self):
        "Getting path of selected folder and passing to loading method."
        model = QStandardItemModel()

        for i, word in enumerate(FILE_NAMES):
            item = QStandardItem(word)
            model.setItem(i, 0, item)

        self.combo.setModel(model)
        self.combo.setModelColumn(0)

    def load_after_getting_path(self, path):
        "Creating instance of Parser class and loading parsed data for our logic."
        self.clipfile_path = path
        self.text.insertPlainText(' ')
        self.text.insertPlainText(
            f'\nClippam s: \n{os.path.basename(path[0])}\n'
        )

    def popup_with_graph(self):
        if self.shapefile:
            self.graph.geometry.plot()
            plt.show()
        else:
            pass

    def popup_clipped_shp(self):
        clipper = Clipper(self.shapefile.path, self.clipfile_path)
        clipper.get_clip_file()
        self.output_gdf = clipper.output_gdf

    def show_table(self):
        if self.output_gdf is not None:
            df = self.output_gdf
            ax = df.plot(column='NKS_KOMB', categorical=True)
            df.apply(lambda x: ax.annotate(s=x['NKS_KOMB'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1);
            plt.show()
        else:
            self.error_dialog = QErrorMessage()
            self.error_dialog.showMessage('Niste clippali!')

    def show_loss_all(self):
        if self.output_gdf is not None:
            self.impact_model = PandasModel(self.calculator.impact_table)
            self.table.setModel(self.impact_model)
            self.table.show()
            self.setup_plotting(self.calculator.impact_table)
            plt.show()
        else:
            self.error_dialog = QErrorMessage()
            self.error_dialog.showMessage('Niste clippali!')

    def show_loss_n2k(self):
        if self.impact_gdf is not None:
            self.calculate_n2k_loss()
            if self.impact_ciljni is not None:
                self.impact_ciljni.drop(
                        columns=['impact_area', 'impact_percentage'],
                        inplace=True,
                        axis=1)
                df = self.impact_ciljni.groupby(
                    ['natura_kod', 'n2k_hab_naziv','ciljni'], as_index=False
                ).sum()
                self.impact_model = PandasModel(df.copy().round(3))
                self.table.setModel(self.impact_model)
                self.table.show()
                self.setup_plotting_n2k(df)
                plt.show()
            else:
                self.error_dialog = QErrorMessage()
                self.error_dialog.showMessage('Nema ciljnih stanista unutar povrsine!')
        else:
            self.error_dialog = QErrorMessage()
            self.error_dialog.showMessage('Niste clippali!')

    def setup_plotting(self, df):
        df = df[['NKS_KOMB', 'impact_percentage', 'impact_area']].sort_values(
            'impact_percentage', ascending=False
        )
        ax = df.plot(x="NKS_KOMB", y=["impact_percentage", "impact_area"], kind="barh")
        ax.set(xlabel="povrsina u ha", ylabel='NKS_KOMB')

    def setup_plotting_n2k(self, df):
        df.reset_index(inplace=True)
        df = df[['natura_kod', 'misjak_min', 'misjak_max']].sort_values(
            'misjak_min', ascending=False
        )
        ax = df.plot(x="natura_kod", y=["misjak_min", "misjak_max"], kind="barh")
        ax.set(xlabel="povrsina u ha", ylabel='natura kod')

    def calculate_loss(self):
        em_gdf = ShapefileLoader(self.shapefile.path).gdf
        self.calculator = ImpactCalculator(em_gdf, self.output_gdf)
        self.calculator.calculate()
        self.impact_gdf = self.calculator.impact_table

    def calculate_n2k_loss(self):
        povs_id_from_path = self.shapefile.path.split('_')[-2]
        self.habitats = HabitatsData(povs_id_from_path, self.impact_gdf)
        self.habitats.get_habitats()
        self.habitats.decode_habitats_to_nks()
        self.impact_ciljni = self.habitats.add_matched_columns()

    def close(self):
        QCoreApplication.instance().quit()


class GetShapeFile:
    def __init__(self, string):
        self.path = None
        for i in ALL_PATHS:
            if string.split()[0] in i:
                self.path = i


class ImpactCalculator:
    def __init__(self, gdf1, gdf2):
        self.gdf_em = gdf1
        self.gdf_clippano = gdf2
        self.impact_table = None

    def calculate(self):

        gdf_clipped = self.gdf_clippano
        gdf = self.gdf_em

        gdf_grouped = gdf.groupby('NKS_KOMB').agg({'AREA_HA': 'sum'})
        gdf_clipped['new_area'] = gdf_clipped['geometry'].area / 10000
        gdf_clipped_grouped = gdf_clipped.groupby('NKS_KOMB').agg({'new_area': 'sum'})
        gdf_clipped_grouped.reset_index(inplace=True)
        gdf_grouped = gdf_grouped.reset_index()

        merged_df = gdf_grouped[['NKS_KOMB', 'AREA_HA']].merge(
            gdf_clipped_grouped[['NKS_KOMB', 'new_area']],
            how='left', on='NKS_KOMB'
        )
        merged_df = merged_df.dropna(subset=['new_area'])
        merged_df['impact_percentage'] = round(
            merged_df['new_area'] / merged_df['AREA_HA'] * 100, 2
        )
        merged_df.rename(
            columns={'new_area': 'impact_area', 'AREA_HA':'total_area'}, inplace=True
        )
        merged_df = merged_df.round(3)
        merged_df = merged_df.sort_values(by = ['impact_percentage'])
        self.impact_table = merged_df

class Clipper:
    def __init__(self, file1, file2):
        self.shapefile = file1
        self.shp_clip = file2[0]
        self.output_gdf = None

    def get_clip_file(self):
        clip_with = geopandas.GeoDataFrame.from_file(self.shp_clip)
        to_clip = geopandas.GeoDataFrame.from_file(self.shapefile)
        gdf_clipped = geopandas.overlay(clip_with, to_clip, how="intersection")
        self.output_gdf = gdf_clipped

class HabitatsData:
    def __init__(self, povs_code, impact_gdf):
        self.povs_code = povs_code
        self.impact_gdf_ciljni = impact_gdf
        self.habitats = None

    def get_habitats(self):
        trazeni_povs = [item for item in DICT_povs if item["natura_id"] == self.povs_code]
        self.habitats = [(item["kod_st_tip"], item['naziv_st_tip']) for item in trazeni_povs]

    def decode_habitats_to_nks(self):
        habitats = self.habitats
        temp_annex = {v: k for k, v in DICT_annex.items()}
        matches = []
        for i in habitats:
            matches.append([(val, key) for key, val in temp_annex.items() if i[0][:3] in key])
            if 'šume' in i[1]:
                matches.append([('E', i[0])])
        matches = [item for sublist in matches for item in sublist]
        return matches

    def add_matched_columns(self):
        matches = self.decode_habitats_to_nks()
        m_d = self.prepare_matches()
        df = self.impact_gdf_ciljni
        hab_dict_cln = dict(self.habitats)
        if any(key.startswith('*')|key.endswith('*') for key in dict(self.habitats)):
            hab_dict_cln = {k.replace('*', ''): v for k, v in (dict(self.habitats)).items()}
        df['ciljni'] = (df['NKS_KOMB']
                        .str.findall(f"({'|'.join([i[0][:3] for i in matches])})", )
                        .str.join(', ')
                        .replace('', np.nan)
                        )
        if not df['ciljni'].isnull().all():
            df = self.parse_special_characters(df, m_d, hab_dict_cln)
            return df
        else:
            df.dropna(inplace=True, axis=1)
            return None


    def parse_special_characters(self, df, m_d, hab_dict_cln):
        df.dropna(inplace=True)
        df['broj_ciljnih'] = df['ciljni'].str.split(",").str.len()
        df = pd.concat(
            [mask_ciljna_stanista(row) for _, row in df.iterrows()],
            ignore_index=True, axis=1
        ).T
        df['ciljni'].replace(' ', '', regex=True, inplace=True)

        if search_dict(m_d, '*'):

            df['natura_kod'] = df['ciljni'].apply(lambda x: [m_d[x]])
            df['natura_kod'] = df['natura_kod'].apply(lambda x: ''.join(x))
            df['natura_kod_temp'] = df['natura_kod'].str.replace('*', '')
            df['n2k_hab_naziv'] = df['natura_kod_temp'].apply(
                lambda x: hab_dict_cln[x])
            df.drop(columns=['natura_kod_temp'], inplace=True)
        else:
            df['natura_kod'] = df['ciljni'].apply(lambda x: m_d[x])
            df['n2k_hab_naziv'] = df['natura_kod'].apply(
                lambda x: hab_dict_cln[x])

        df['temp'] = df['NKS_KOMB'].str.split(' ')
        df['misjak_min'] = df.apply(f_min, axis=1)
        df['misjak_max'] = df.apply(f_max, axis=1)
        df.drop(columns=['temp','broj_ciljnih'], axis=1, inplace=True)
        return df.round(3)

    def prepare_matches(self):
        matches = self.decode_habitats_to_nks()
        matches_dict = dict(matches)
        return {k[:3]: v for k, v in matches_dict.items()}


class PandasModel(QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class ExtendedCombo(QComboBox):
    def __init__(self, parent=None):
        super(ExtendedCombo, self).__init__(parent)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setEditable(True)
        # add a filter model to filter matching items
        self.pFilterModel = QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())
        # add a completer, which uses the filter model
        self.completer = QCompleter(self.pFilterModel, self)
        # always show all (filtered) completions
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)
        # connect signals
        self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)
        self.cur_index = self.currentIndex()
        self.text = self.currentText()

    # on selection of an item from the completer, select the corresponding item from combobox
    def on_completer_activated(self, text):
        if text:
            index = self.findText(text)
            self.setCurrentIndex(index)
            C = TextInstance
            C.text = text
        else:
            C = TextInstance
            C.text = self.currentText()

    def update(self):
        return self.text

    # on model change, update the models of the filter and completer as well
    def setModel(self, model):
        super(ExtendedCombo, self).setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    # on model column change, update the model column of the filter and completer as well
    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super(ExtendedCombo, self).setModelColumn(column)


class TextInstance():
    text = 'bla'
    def __init__(self, value):
        self.text = value


class ShapefileLoader:
    def __init__(self, path):
        self.gdf = geopandas.read_file(path)


def search_dict(myDict, lookup):
    for key, value in myDict.items():
        for v in value:
            if lookup in v:
                return key

stylesheet = """
        Window {
        border-image: url("background.jpg"); 
        background-repeat: no-repeat; 
        background-position: center;
    }
"""

App = QApplication(sys.argv)
App.setStyleSheet(stylesheet)
App.setFont(QFont('Roboto', 9))
window = Window()
sys.exit(App.exec())
