
class CMIP6_date_tools:

    def define_dpm(self):
        dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
               '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
               'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
               'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
               'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
               'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
               '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
               '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}
        return dpm


    def leap_year(self, year, calendar='standard'):
        """Determine if year is a leap year"""
        leap = False
        if ((calendar in ['standard', 'gregorian',
                          'proleptic_gregorian', 'julian']) and
                (year % 4 == 0)):
            leap = True
            if ((calendar == 'proleptic_gregorian') and
                    (year % 100 == 0) and
                    (year % 400 != 0)):
                leap = False
            elif ((calendar in ['standard', 'gregorian']) and
                  (year % 100 == 0) and (year % 400 != 0) and
                  (year < 1583)):
                leap = False
        return leap


    def get_dpm(self, time, calendar='standard'):
        """
        return a array of days per month corresponding to the months provided in `months`
        """
        month_length = np.zeros(len(time), dtype=np.int)

        dpm = self.define_dpm()
        cal_days = dpm[calendar]

        for i, (month, year) in enumerate(zip(time.month, time.year)):
            month_length[i] = cal_days[month]
            if self.leap_year(year, calendar=calendar) and month == 2:
                month_length[i] += 1
        return month_length