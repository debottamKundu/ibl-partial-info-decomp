import pytest

from ibl_info.utility import aggregated_regions_time_intervals, aggregated_regions_time_resolved
import numpy as np

def test_aggregated_regions_time_intervals():
    input_data = np.asarray([[5,5,2],[6,0,1],[3,3,8]]) # trials x neurons
    region_info = np.asarray(['CA1','CA3','CA1'])
    expected_output = np.asarray([[7,5],[7,0],[11,3]]) # traisl x regions
    
    output, regions_output = aggregated_regions_time_intervals(input_data, region_info)

    np.testing.assert_allclose(output, expected_output)


    