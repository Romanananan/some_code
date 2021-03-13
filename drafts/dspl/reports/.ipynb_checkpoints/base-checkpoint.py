from typing import Optional
import numpy as np
import pandas as pd


class BaseReport:
    """
    Набор правил и стилей для форматирования отчета.

    Parameters
    ----------
    saver: dspl.utils.INFOSaver
        экземпляр сохранения выходных файлов

    """
    def __init__(self, saver):
        self.dir = saver.dir_
        self.writer = pd.ExcelWriter(path=f"{self.dir}/docs/report.xlsx")
        self.sheets = self.writer.sheets
        self.wb = self.writer.book
        self.predictions = None

    def add_table_borders(self,
                          data,
                          sheet_name: str,
                          startrow: int = 0,
                          num_format: int = 10):
        """
        Установка границ таблицы на листе Excel.

        Parameters
        ----------
        data: pandas.DataFrame
            Набор данных для записи в Excel.

        sheet_name: string
            Название листа в Excel-книге для записи данных.

        startrow: integer, optional, default = 0
            Номер строки, с которой начинать запись данных.

        num_format: integer, optional, default = 10
            Числовой формат записи данных.

        """
        ws = self.sheets[sheet_name]
        last_col = data.columns[-1]

        # запись последнего столбца в data
        if num_format:
            sheet_format = self.wb.add_format(
                {"right": 1, "num_format": num_format})
        else:
            sheet_format = self.wb.add_format({"right": 1})

        for cell_number, data_value in enumerate(data[last_col]):
            row_idx, col_idx = startrow + cell_number + 1, data.shape[1] - 1
            ws.write(row_idx, col_idx, data_value, sheet_format)

        # запись последней строки в data
        sheet_format = self.wb.add_format({"bottom": 1})
        for cell_number, data_value in enumerate(data.values[-1]):
            row_idx, col_idx = startrow + data.shape[0], cell_number
            ws.write(row_idx, col_idx, data_value, sheet_format)

        # запись элемента последней строки и последнего столбца в data
        if num_format:
            sheet_format = self.wb.add_format(
                {"right": 1, "bottom": 1, "num_format": num_format})
        else:
            sheet_format = self.wb.add_format({"right": 1, "bottom": 1})

        row_idx, col_idx = startrow + data.shape[0], data.shape[1] - 1
        ws.write(row_idx, col_idx, data.values[-1, -1], sheet_format)

    def add_cell_width(self, data, sheet_name: str):
        """
        Установка ширины ячейки на листе Excel.

        Parameters
        ----------
        data: pandas.DataFrame
            Набор данных для записи в Excel.

        sheet_name: string
            Название листа в Excel-книге для записи данных.

        """
        ws = self.sheets[sheet_name]
        for cell_number, table_column in enumerate(data.columns):
            max_value_len = data[table_column].astype("str").str.len().max()
            cell_len = max(max_value_len, len(table_column)) + 2
            ws.set_column(cell_number, cell_number, cell_len)

    def add_header_color(self,
                         data,
                         sheet_name: str,
                         startrow: int = 0,
                         color: str = "77d496"):
        """
        Установка цвета заголовка на листе Excel.

        Parameters
        ----------
        data: pandas.DataFrame
            Набор данных для записи в Excel.

        sheet_name: string
            Название листа в Excel-книге для записи данных.

        startrow: integer, optional, default = 0
            Номер строки, с которой начинать запись данных.

        color: string, optional, default = "77d496"
            RGB-цвет заголовка.

        """
        ws = self.sheets[sheet_name]
        sheet_format = self.wb.add_format({
            "bold": True, "text_wrap": True,
            "fg_color": color, "border": 1,
            "align": "center"})

        for cell_number, data_value in enumerate(data.columns.values):
            ws.write(startrow, cell_number, data_value, sheet_format)

    def add_numeric_format(self,
                           data,
                           sheet_name: str,
                           startrow: int = 0,
                           startcol: int = 0,
                           min_value: int = 10,
                           color: str = "#000000"):
        """
        Установка формата для числовой таблицы.

        Parameters
        ----------
        data: pandas.DataFrame
            Набор данных для записи в Excel.

        sheet_name: string
            Название листа в Excel-книге для записи данных.

        startrow: integer, optional, default = 0
            Номер строки, с которой начинать запись данных.

        startcol: integer, optional, default = 0
            Номер столбца, с которого начинать запись данных.

        set_top: integer, optional, default = 0

        min_value: interger, optional, default = 10
            Минимальное значение, формат записи которого "х",
            если значение в ячейке Excel-книге меньше min_value,
            то формат записи - "x.yy".

        color: string, optional, default = "#000000"
            RGB-цвет шрифта.

        """
        ws = self.sheets[sheet_name]

        for col_number, column in enumerate(data.columns[1:]):
            if col_number == data.shape[1] - 2:
                fmt = {"right": 1}
            else:
                fmt = {}

            for row_number, value in enumerate(data[column]):
                try:
                    if row_number == data.shape[0] - 1:
                        fmt.update({"bottom": 1, "font_color": color})

                    if np.abs(value) > min_value or np.abs(value) in range(min_value):
                        fmt.update({"num_format": 1, "font_color": color})
                        sheet_format = self.wb.add_format(fmt)
                    elif np.abs(value) <= min_value:
                        fmt.update({"num_format": 2, "font_color": color})
                        sheet_format = self.wb.add_format(fmt)

                except np.core._exceptions.UFuncTypeError:
                    fmt = {"right": 1}
                    sheet_format = self.wb.add_format(fmt)

                ws.write(
                    startrow + row_number + 1,
                    startcol + col_number + 1,
                    value, sheet_format
                )

    def add_text_color(self,
                       sheet_name: str,
                       startcol: int,
                       endcol: int,
                       color: str = "C8C8C8"):
        """
        Добавление отдельного цвета текста на на листе Excel.

        Parameters
        ----------
        sheet_name: string
            Название листа в Excel-книге для записи данных.

        startcol: integer
            Номер столбца, с которого начинать запись данных.

        endcol: integer
            Номер столбца, на котором закончить запись данных.

        color: string, optional, default = "C8C8C8"
            RGB-цвет шрифта.

        """
        ws = self.sheets[sheet_name]
        cols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        startcol, endcol = cols[startcol], cols[endcol]

        sheet_format = self.wb.add_format({"font_color": color})
        ws.set_column(f"{startcol}:{endcol}", None, sheet_format)

    def add_eventrate_format(self,
                             data,
                             sheet_name: str,
                             startcol: int = 4,
                             fmt: int = 10):
        """
        Добавление формата для eventrate на листах Excel.

        Parameters
        ----------
        data: pandas.Series
            Столбец со значениями eventrate.

        sheet_name: string
            Название листа в Excel-книге для записи данных.

        startcol: integer, optional, default = 4
            Номер столбца, в котором требуется установить формат.
            Опциональный параметр, по умолчанию используется стобец 4.

        fmt: integer, optional, default = 10
            Код формата xlsxwriter.

        """
        ws = self.sheets[sheet_name]
        sheet_format = self.wb.add_format({"num_format": fmt})

        for cell_number, data_value in enumerate(data):
            ws.write(1 + cell_number, startcol, data_value, sheet_format)

        sheet_format = self.wb.add_format({"num_format": fmt, "bottom": True})
        ws.write(len(data), startcol, data_value, sheet_format)

    def add_bottom_table_borders(self, data, sheet_name: str, startrow: int = 0):
        """
        Установка верхней и нижней границ таблицы на листе Excel.

        Parameters
        ----------
        data: pandas.Series
            Столбец со значениями таблицы.

        sheet_name: string
            Название листа в Excel-книге для записи данных.

        startrow: integer, optional, default = 0
            Номер строки, с которой начинать запись данных.

        """
        ws = self.sheets[sheet_name]

        for cell_number, data_value in enumerate(data):
            if isinstance(data_value, str):
                fmt = {
                    "bottom": 1, "left": 1, "right": 1,
                    "top": 1, "bold": True
                }
            elif data_value > 100:
                fmt = {
                    "bottom": 1, "left": 1, "right": 1,
                    "top": 1, "num_format": 1, "bold": True
                }
            elif data_value < 100 and cell_number != 4:
                fmt = {
                    "bottom": 1, "left": 1, "right": 1,
                    "top": 1, "num_format": 2, "bold": True
                }
            elif cell_number == 4:
                fmt = {
                    "bottom": 1, "left": 1, "right": 1,
                    "top": 1, "num_format": 10, "bold": True
                }

            sheet_format = self.wb.add_format(fmt)
            ws.write(startrow, cell_number, data_value, sheet_format)

    def create_compare_models_url(self, data, sheet_name: str):
        """
        Создание ссылки на лист Compare Models и добавление
        на лист с отчетом для пары модель / выборка.

        Parameters
        ----------
        data: pandas.DataFrame
            Набор данных для записи в Excel.

        sheet_name: string
            Название листа в Excel-книге для записи данных.

        """
        ws = self.sheets[sheet_name]
        string = "Ссылка на лист сравнения моделей"
        url = "internal:'Compare Models'!A1"

        df = data.loc[max(data.index)]
        ws.write_url(f"A{len(data)+2}", url, string=string)
        self.add_bottom_table_borders(df, sheet_name, data.shape[0])
        self.add_cell_width(data, sheet_name)

    def create_model_url(self, **eval_sets):
        """
        Создание ссылки на лист с отчетом модель / выборка и
        добавление на лист Compare Models.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        ws = self.sheets["Compare Models"]
        cols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        sheet_format = self.wb.add_format({"right": True})
        train_sheets = [sheet for sheet in self.sheets if "train" in sheet]

        for sheet_number, sheet_name in enumerate(train_sheets):
            url = f"internal:'{sheet_name}'!A1"
            string = f"Ссылка на лист {sheet_name}"
            try:
                cell_name = cols[2 + 3*len(eval_sets)]
            except IndexError:
                n_letter = abs(len(cols) - 6*len(eval_sets) - 2)
                cell_name = "A{}".format(cols[n_letter])

            ws.write_url(f"{cell_name}{sheet_number + 2}", url, sheet_format, string)

        sheet_format = self.wb.add_format({"right": True, "bottom": True})
        ws.write_url(f"{cell_name}{sheet_number + 2}", url, sheet_format, string)

    def set_style(self,
                  data,
                  sheet_name: str,
                  startrow: int = 0,
                  color: str = "77d496",
                  num_format: Optional[int] = None):
        """
        Установка базового стиля для всех листов Excel-книги.
        Базовый стиль включает в себя:
            - установку границ таблицы;
            - установку оптимального размера ячейки;
            - установку цвета заголовка таблицы;
            - форматирование шрифтов.

        Parameters
        ----------
        data: pandas.DataFrame
            Набор данных для записи в Excel.

        sheet_name: string
            Название листа в Excel-книге для записи данных.

        startrow: integer, optional, default = 0
            Номер строки, с которой начинать запись данных.

        color: string, optional, default = "77d496"
            RGB-цвет заголовка.

        num_format: integer, optional, default = None
            Числовой формат записи данных.

        """
        self.add_table_borders(data, sheet_name, startrow, num_format)
        self.add_header_color(data, sheet_name, startrow, color)
        self.add_cell_width(data, sheet_name)


def create_used_features_stats(data: pd.DataFrame, models: dict):
    """
    Создание датафрейма со списком используемых признаков для каждой
    модели. Если признак используется в модели - маркируется 1,
    иначе - маркируется 0.

    Parameters
    ----------
    data: pandas.DataFrame, shape = [n_features, 3]
        Датафрейм с оценкой важности признаков.

    models: dict
        Словарь с экземплярами моделей.

    Returns
    -------
    df: pandas.DataFrame, shape = [n_features, len(models)]
        Датафрейм с флагами использования признаков.

    """
    df = pd.DataFrame(
        {"Variable": data["Variable"]}
    )

    for model in models:
        used_features = models[model].used_features
        df[f"{model} include variable"] = df["Variable"].isin(used_features).astype(int)

    return df
