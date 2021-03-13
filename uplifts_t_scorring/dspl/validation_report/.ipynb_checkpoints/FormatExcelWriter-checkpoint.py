import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell
from typing import Tuple

class FormatExcelWriter:
    """
    Класс реализации интерфейса записи датафреймов в excel книгу с применением
    форматирования

    Parameters:
    -----------
    writer: pd.ExcelWriter
        Объект excel-writer для записи. Подразумевается что excel файл был
        создан вне класса
    """

    def __init__(self, writer: pd.ExcelWriter = None):
        self.writer = writer
        self.workbook = self.writer.book
        self.formats = {}

        # Add a header format.
        self.header_format = self.workbook.add_format({
            'bold': True,
            'text_wrap': False,
            'valign': 'top',
            'fg_color': '#BFBFBF',
            'center_across': True,
            'border': 1})

        # Наборы форматов для условного форматирования и светофора
        # Green fill with dark green text.
        self.format_green = self.workbook.add_format({'bg_color': '#C6EFCE',
                                                      'font_color': '#006100'})
        # Light red fill with dark red text.
        self.format_red = self.workbook.add_format({'bg_color': '#FFC7CE',
                                                    'font_color': '#9C0006'})
        #  Light yellow fill with brown text.
        self.format_yellow = self.workbook.add_format({'bg_color': '#FFFFCC',
                                                       'font_color': '#CC9900'})

    def _print_header(self, df: pd.DataFrame, pos: tuple, sheet: str):
        """
        Вывод заголовка таблицы в определенном формате по указанному
        расположению

        Parameters:
        -----------
        df: pd.DataFrame
            Датафрейм, заголовок которого нужно вывести
        pos: tuple
            Кортеж с координтаами ячейки для вывода заголовка
        """
        rows, cols = pos

        # Записать элементы заголовка    
        for col_num, value in enumerate(df.columns.values):
            self.worksheet.write(rows, cols + col_num, value,
                                 self.header_format)

    def _print_data(self, df: pd.DataFrame, pos: tuple, sheet: str):
        """
        Запись элементов датафреймам на лист excel

        Parameters:
        -----------
        df: pd.DataFrame
            Датафрейм для записи
        pos: tuple
            кортеж с координатами левого верхнего угла таблицы
        """
        rows, cols = pos

        for row_num, row in enumerate(df.values):
            for col_num, val in enumerate(row):
                # xlsxwriter не может записывать список
                # только str или numeric
                if str(type(val)).endswith("'list'>"):
                    val = "".join(val)

                if pd.isnull(val):
                    print_val = " "
                else:
                    print_val = val

                cell = xl_rowcol_to_cell(rows + row_num + 1, cols + col_num)

                # Если есть формат для ячейки - применить его
                if cell in self.formats.keys():
                    fmt = self.workbook.add_format(self.formats[cell])
                    self.worksheet.write(cell, print_val, fmt)
                else:
                    self.worksheet.write(cell, print_val)

    def _reset_format(self):
        """
        Восстановлнение словаря форматов для таблицы,
        необходимо после успешного применения формата
        """
        self.formats = {}

    def merge_cells(self, df: pd.DataFrame, pos: Tuple, col_start: str,
                    col_end: str, row_start: int, row_end: int):
        """
        Объединение ячеек на листе excel
        """

        merge_format = self.workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'font_color': "#BFBFBF",
            'text_wrap': True
        })

        row, col = pos
        first_col = df.columns.get_loc(col_start) + col
        last_col = df.columns.get_loc(col_end) + col

        if row_start == 0:
            merge_format.set_top(5)
        if row_end == df.shape[0]-1:
            merge_format.set_bottom(5)
        if first_col == 0:
            merge_format.set_left(5)
        if last_col == df.shape[1]-1:
            merge_format.set_right(5)

        self.worksheet.merge_range(first_row=row_start + row+1,
                                   first_col=first_col,
                                   last_row=row_end + row+1,
                                   last_col=last_col,
                                   data=df[col_start][row_start],
                                   cell_format=merge_format
                                   )

    def _add_cell_format(self, cell: str, format_name, format_value):
        """
        Добавление к словарю форматов формат для конкретной ячейки

        Parameters:
        -----------
        cell:str
            Имя ячейки
        format_name: str
            Наимнования аттрибута формата
        format_value:
            значние аттрибута формата

        """
        format_dict = {format_name: format_value}

        if cell in self.formats.keys():
            tmp = self.formats[cell]
            tmp[format_name] = format_value
            self.formats[cell] = tmp
        else:
            self.formats[cell] = {format_name: format_value}

    def _add_column_format(self, df: pd.DataFrame, pos: tuple,
                           col_name: str, format_name, format_value):
        """
        Добавление формата для каждой ячейки столбца

        Parameters:
        -----------
        df: pd.DataFrame
            Датафрейм для которого форматируется столбец
        pos: tuple
            Координаты левого верхнего угла таблицы на листе
        col_name: str
            Имя столбца для форматирования
        format_name: str
            Наименование устанавлилваемого формата
        format_value
            Значение устанавливаемого формата

        """
        rows, cols = pos
        col_num = df.columns.get_loc(col_name)

        for row_num in range(df.shape[0]):
            cell = xl_rowcol_to_cell(rows + row_num + 1, cols + col_num)
            self._add_cell_format(cell, format_name, format_value)

    def _add_row_format(self, df: pd.DataFrame, pos: tuple,
                        row_num: int, format_name, format_value):
        """
        Добавление формата для каждой ячейки строки

        Parameters:
        -----------
        df: pd.DataFrame
            Датафрейм для которого форматируется строка
        pos: tuple
            Координаты левого верхнего угла таблицы на листе
        row_num: str
            Номер строки для форматирования
        format_name: str
            Наименование устанавлилваемого формата
        format_value
            Значение устанавливаемого формата
        """
        rows, cols = pos

        for col_num in range(df.shape[1]):
            cell = xl_rowcol_to_cell(rows + row_num + 1, cols + col_num)
            self._add_cell_format(cell, format_name, format_value)

    def _add_bold_border(self, df: pd.DataFrame, pos: tuple, thickness: int):
        """
        Добавляет форматы для граничных ячеек с установкой толстых границ

        Parameters:
        -----------
        df: pd.DataFrame
            Датафрейм, для которого добалвяется жирная внешняя граница
        pos:tuple
            Координаты левого верхнего угла таблицы на странице excel
        thickness:int
            Толщина внешней границы
        """
        rows, cols = pos
        # Верхняя-нижняя граница
        for col_num in range(df.shape[1]):
            top_cell = xl_rowcol_to_cell(rows + 1, cols + col_num)
            self._add_cell_format(top_cell,
                                  format_name="top", format_value=thickness)

            bottom_cell = xl_rowcol_to_cell(rows + df.shape[0], cols + col_num)
            self._add_cell_format(bottom_cell,
                                  format_name="bottom", format_value=thickness)

        # Левая-правая граница
        for row_num in range(df.shape[0]):
            left_cell = xl_rowcol_to_cell(rows + row_num + 1, cols)
            self._add_cell_format(left_cell,
                                  format_name="left", format_value=thickness)

            right_cell = xl_rowcol_to_cell(rows + row_num + 1,
                                           cols + df.shape[1] - 1)
            self._add_cell_format(right_cell,
                                  format_name="right", format_value=thickness)

    def set_width(self, df: pd.DataFrame, width: int, pos: tuple):
        rows, cols = pos
        for num, col in enumerate(df.columns):
            self.worksheet.set_column(cols + num, cols + num, width)

    def set_height(self, df: pd.DataFrame, height: int, pos: tuple):
        rows, cols = pos
        for row in range(df.shape[0]):
            self.worksheet.set_row(row + rows + 1, height)

    def _set_cells_width(self, df: pd.DataFrame, pos: tuple, sheet: str):
        """
        Установка ширины ячейки листа в Excel.
        Для расчета ширины ячейки вычисляется максимальная длина
        значения в столбце data и длина заголовка столбца data, и
        выбирается максимальная длина из этих двух значений.

        Parameters:
        -----------
        df: pandas.DataFrame
            матрица для записи данных.
        pos:tuple
            Координаты левого верхнего угла таблицы на странице excel
        sheet: string
            название листа.
        """
        rows, cols = pos
        for num, column in enumerate(df.columns):
            column_len = df[column].astype("str").str.len().max()
            column_len = max(column_len, len(column)) + 2
            self.worksheet.set_column(num + cols, num + cols, column_len)

    @staticmethod
    def define_auto_formats(df: pd.DataFrame) -> dict:
        """
        Автоматическое определение формата вывода таблицы по установленному
        принципу:
            Для типа целых чисел: "## ##0"
            Для типа float с макс. значением больше 10 : "## ##0.00"
            Для типа float с макс. значением менее 10 : "## ##0.0000"

        Parameters:
        -----------
        df:pd.DataFrame
            Исходный датафрейм для установки форматов
        """

        # Базовые форматы вывода
        int_number = "## ##0"
        float_number_high = "## ##0.00"
        float_number_low = "## ##0.0000"
        int_percentage = "0%"
        float_percentage = "0.00%"

        # types
        integers = ['int16', 'int32', 'int64']
        floats = ['float16', 'float32', 'float64']

        high_floats = []
        low_floats = []
        ints = []

        for col in df.columns:
            if df[col].dtype in floats and df[col].max() > 10:
                high_floats.append(col)
            elif df[col].dtype in floats and df[col].max() <= 10:
                low_floats.append(col)
            elif df[col].dtype in integers:
                ints.append(col)

        formats = {"num_format": {int_number: ints,
                   float_number_high: high_floats,
                   float_number_low: low_floats,
                   int_percentage: [],
                   float_percentage: []}}
        return formats

    def set_cell_cond_format(self, df:pd.DataFrame, pos:tuple, col_name:str,
                             row_num: int, upper, lower, order:str):
        """
        Установка условного форматирования на конкретную ячейку в excel, куда
        записался датафрейм
        """
        rows, cols = pos

        col_num = df.columns.get_loc(col_name)

        fmt_start = xl_rowcol_to_cell(rows+row_num+1, cols + col_num)
        fmt_end = xl_rowcol_to_cell(rows+row_num+1, cols + col_num)

        if order == "straight":
            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "between",
                                               "minimum": lower,
                                               "maximum": upper,
                                               "format": self.format_yellow})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "<",
                                               "value": lower,
                                               "format": self.format_green})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": ">",
                                               "value": upper,
                                               "format": self.format_red})

        elif order == "reverse":
            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "between",
                                               "minimum": upper,
                                               "maximum": lower,
                                               "format": self.format_yellow})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": ">",
                                               "value": lower,
                                               "format": self.format_green})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "<",
                                               "value": upper,
                                               "format": self.format_red})


    def set_col_cond_format_tail(self, df: pd.DataFrame, pos: tuple,
                                 col_name: str, upper, lower, order: str):
        """
        Утановка условного форматирования на ячейки таблицы. Односторонее
        форматирование

        Parameters:
        -----------
        df:pd.DataFrame
            исходный датафрейм
        pos:tuple
            Координаты левого верхнего угла таблицы на странице excel
        col_name:
            имя столбца для установки формата
        upper
            верхняя граница УФ
        lower
            нижняя граница УФ
        order:str
            order = "straight":
                        < lower - green
                        >= upper - red

            order = "reverse":
                        > lower - green
                        <= upper - red
        """
        rows, cols = pos
        col_num = df.columns.get_loc(col_name)

        fmt_start = xl_rowcol_to_cell(rows + 1, cols + col_num)
        fmt_end = xl_rowcol_to_cell(rows + df.shape[0], cols + col_num)

        if order == "straight":
            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "<",
                                               "value": lower,
                                               "format": self.format_green})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": ">=",
                                               "value": upper,
                                               "format": self.format_red})

        elif order == "reverse":
            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": ">",
                                               "value": lower,
                                               "format": self.format_green})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "<=",
                                               "value": upper,
                                               "format": self.format_red})

    def set_col_cond_format(self, df: pd.DataFrame, pos: tuple,
                            col_name: str, upper, lower, order: str):
        """
        Утановка условного форматирования на ячейки таблицы. двустороннее
        форматирование

        Parameters:
        -----------
        df:pd.DataFrame
            исходный датафрейм
        pos:tuple
            Координаты левого верхнего угла таблицы на странице excel
        col_name:
            имя столбца для установки формата
        upper
            верхняя граница УФ
        lower
            нижняя граница УФ
        order:str

            order = "straight":
                    < lower - green
                    lower-upepr - yellow
                    > upper - red

            order = "reverse":
                    > lower - green
                    lower-upper - yellow
                    < upper - red
        """
        rows, cols = pos
        col_num = df.columns.get_loc(col_name)

        fmt_start = xl_rowcol_to_cell(rows + 1, cols + col_num)
        fmt_end = xl_rowcol_to_cell(rows + df.shape[0], cols + col_num)

        if order == "straight":
            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "between",
                                               "minimum": lower,
                                               "maximum": upper,
                                               "format": self.format_yellow})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "<",
                                               "value": lower,
                                               "format": self.format_green})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": ">",
                                               "value": upper,
                                               "format": self.format_red})

        elif order == "reverse":
            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "between",
                                               "minimum": upper,
                                               "maximum": lower,
                                               "format": self.format_yellow})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": ">",
                                               "value": lower,
                                               "format": self.format_green})

            self.worksheet.conditional_format(f"{fmt_start}:{fmt_end}",
                                              {"type": "cell",
                                               "criteria": "<",
                                               "value": upper,
                                               "format": self.format_red})

    def write_data_frame(self,
                         df: pd.DataFrame,
                         pos: tuple,
                         sheet: str,
                         formats: dict = None,
                         row_formats: dict = None):
        """
        Вызов интерфейса записи в excel файл.
        Записывает на указанный лист в указаную позицию таблицу.
        Применяет указанные форматы к столбцам и строкам.

        df:pd.DataFrame
            исходный датафрейм для записи
        pos:tuple
            Координаты левого верхнего угла таблицы на странице excel
        sheet:str
            Имя листа
        formats:dict
            словарь форматов для столбцов
        row_formas:dict
            словарь формтаов для строк
        """
        # excel sheet name length limitation
        worksheet = sheet[:31]
        df_empty = pd.DataFrame()
        # Записать пустой датафрейм в excel файл для корректного создания sheet
        df_empty.to_excel(excel_writer=self.writer,
                          sheet_name=worksheet,
                          startrow=0,
                          index=False,
                          header=False,
                          engine='OpenPyXL')

        self.worksheet = self.writer.sheets[worksheet]

        # Вывести отформатиованный заголовок таблицы
        self._print_header(df=df
                           , pos=pos
                           , sheet=worksheet)
        if formats is None:
            formats = self.define_auto_formats(df)

        # Добавить форматы для столбцов
        for fmt_name, fmt in formats.items():
            for value, cols in fmt.items():
                for col in cols:
                    if col in df.columns:
                        self._add_column_format(df=df,
                                                pos=pos,
                                                col_name=col,
                                                format_name=fmt_name,
                                                format_value=value)
        if row_formats is not None:
            # Добавить форматы для строк
            for fmt_name, fmt in row_formats.items():
                for value, rows in fmt.items():
                    for row in rows:
                        self._add_row_format(df=df,
                                             pos=pos,
                                             row_num=row,
                                             format_name=fmt_name,
                                             format_value=value)

        # Добавить жирную границу вокруг таблицы
        self._add_bold_border(df=df,
                              pos=pos,
                              thickness=5)

        # Вывести отформатированную таблицу
        self._print_data(df=df,
                         pos=pos,
                         sheet=worksheet)

        #  Установить ширину столбцов
        self._set_cells_width(df=df, pos=pos, sheet=worksheet)

        self._reset_format()

