#!/usr/bin/env python
"""Basic test of the Calc module on 2D data."""
import datetime
from os.path import isfile
import shutil
import unittest

import xarray as xr

from aospy.calc import Calc, CalcInterface, _add_units_and_description_attrs
from .data.objects.examples import (
    example_proj, example_model, example_run, condensation_rain,
    precip, sphum, globe, sahel
)


class TestCalcBasic(unittest.TestCase):
    def setUp(self):
        self.test_params = {
            'proj': example_proj,
            'model': example_model,
            'run': example_run,
            'var': condensation_rain,
            'date_range': (datetime.datetime(4, 1, 1),
                           datetime.datetime(6, 12, 31)),
            'intvl_in': 'monthly',
            'dtype_in_time': 'ts'
        }

    def tearDown(self):
        for direc in [example_proj.direc_out, example_proj.tar_direc_out]:
            shutil.rmtree(direc)

    def test_annual_mean(self):
        calc_int = CalcInterface(intvl_out='ann',
                                 dtype_out_time='av',
                                 **self.test_params)
        calc = Calc(calc_int)
        calc.compute()
        assert isfile(calc.path_out['av'])
        assert isfile(calc.path_tar_out)
        _test_output_attrs(calc, 'av')

    def test_annual_ts(self):
        calc_int = CalcInterface(intvl_out='ann',
                                 dtype_out_time='ts',
                                 **self.test_params)
        calc = Calc(calc_int)
        calc.compute()
        assert isfile(calc.path_out['ts'])
        assert isfile(calc.path_tar_out)

    def test_seasonal_mean(self):
        calc_int = CalcInterface(intvl_out='djf',
                                 dtype_out_time='av',
                                 **self.test_params)
        calc = Calc(calc_int)
        calc.compute()
        assert isfile(calc.path_out['av'])
        assert isfile(calc.path_tar_out)

    def test_seasonal_ts(self):
        calc_int = CalcInterface(intvl_out='djf',
                                 dtype_out_time='ts',
                                 **self.test_params)
        calc = Calc(calc_int)
        calc.compute()
        assert isfile(calc.path_out['ts'])
        assert isfile(calc.path_tar_out)

    def test_monthly_mean(self):
        calc_int = CalcInterface(intvl_out=1,
                                 dtype_out_time='av',
                                 **self.test_params)
        calc = Calc(calc_int)
        calc.compute()
        assert isfile(calc.path_out['av'])
        assert isfile(calc.path_tar_out)

    def test_monthly_ts(self):
        calc_int = CalcInterface(intvl_out=1,
                                 dtype_out_time='ts',
                                 **self.test_params)
        calc = Calc(calc_int)
        calc.compute()
        assert isfile(calc.path_out['ts'])
        assert isfile(calc.path_tar_out)

    def test_simple_reg_av(self):
        calc_int = CalcInterface(intvl_out='ann',
                                 dtype_out_time='reg.av',
                                 region=[globe],
                                 **self.test_params)
        calc = Calc(calc_int)
        calc.compute()
        assert isfile(calc.path_out['reg.av'])
        assert isfile(calc.path_tar_out)
        _test_output_attrs(calc, 'reg.av')

    def test_simple_reg_ts(self):
        calc_int = CalcInterface(intvl_out='ann',
                                 dtype_out_time='reg.ts',
                                 region=[globe],
                                 **self.test_params)
        calc = Calc(calc_int)
        calc.compute()
        assert isfile(calc.path_out['reg.ts'])
        assert isfile(calc.path_tar_out)

    def test_complex_reg_av(self):
        calc_int = CalcInterface(intvl_out='ann',
                                 dtype_out_time='reg.av',
                                 region=[sahel],
                                 **self.test_params)
        calc = Calc(calc_int)
        calc.compute()
        assert isfile(calc.path_out['reg.av'])
        assert isfile(calc.path_tar_out)

class TestCalcComposite(TestCalcBasic):
    def setUp(self):
        self.test_params = {
            'proj': example_proj,
            'model': example_model,
            'run': example_run,
            'var': precip,
            'date_range': (datetime.datetime(4, 1, 1),
                           datetime.datetime(6, 12, 31)),
            'intvl_in': 'monthly',
            'dtype_in_time': 'ts'
        }


class TestCalc3D(TestCalcBasic):
    def setUp(self):
        self.test_params = {
            'proj': example_proj,
            'model': example_model,
            'run': example_run,
            'var': sphum,
            'date_range': (datetime.datetime(6, 1, 1),
                           datetime.datetime(6, 1, 31)),
            'intvl_in': 'monthly',
            'dtype_in_time': 'ts',
            'dtype_in_vert': 'sigma',
            'dtype_out_vert': 'vert_int'
        }

class TestCalcAttrs(unittest.TestCase):
    def test_units_description_attrs(self):
        _test_attrs('', '', None, '', '')
        _test_attrs('m', '', None, 'm', '')
        _test_attrs('', 'rain', None, '', 'rain')
        _test_attrs('m', 'rain', None, 'm', 'rain')
        _test_attrs('', '', 'vert_av', '', '')
        _test_attrs('m', '', 'vert_av', 'm', '')
        _test_attrs('', 'rain', 'vert_av', '', 'rain')
        _test_attrs('m', 'rain', 'vert_av', 'm', 'rain')
        _test_attrs('', '', 'vert_int',
                    '(vertical integral of quantity with unspecified units)',
                    '')
        _test_attrs('m', '', 'vert_int',
                    '(vertical integral of m): m kg m^-2)',
                    '')
        _test_attrs('', 'rain', 'vert_int',
                    '(vertical integral of quantity with unspecified units)',
                    'rain')
        _test_attrs('m', 'rain', 'vert_int',
                    '(vertical integral of m): m kg m^-2)',
                    'rain')

def _test_attrs(units, description, dtype_out_vert, expected_units,
                expected_description):
    da = xr.DataArray(None)
    ds = xr.Dataset({'bar': 'foo', 'boo': 'baz'})
    _add_units_and_description_attrs(da, units, description, dtype_out_vert)
    _add_units_and_description_attrs(ds, units, description, dtype_out_vert)
    assert expected_units == da.attrs['units']
    assert expected_description == da.attrs['description']
    for name, d_arr in ds.data_vars.items():
        assert expected_units == d_arr.attrs['units']
        assert expected_description == d_arr.attrs['description']

def _test_output_attrs(calc, file):
    data = xr.open_dataset(calc.path_out[file])
    expected_units = calc.var.units
    if calc.dtype_out_vert == 'vert_int':
        if expected_units != '':
            expected_units = ("(vertical integral of {0}):"
                              " {0} m)").format(expected_units)
        else:
            expected_units = ("(vertical integral of quantity"
                              " with unspecified units)")
    expected_description = calc.var.description
    for name, da in data.data_vars.items():
        assert expected_units == da.attrs['units']
        assert expected_description == da.attrs['description']

if __name__ == '__main__':
    unittest.main()
