'''
Update selective indexing years based on the coverage note text
'''
from datetime import date
import re
import xml.etree.ElementTree as ET

CITATION_SUBSETS = [
    'AIM', 
    'B', 
    'C', 
    'D', 
    'E', 
    'F', 
    'H', 
    'IM', 
    'J', 
    'K', 
    'N', 
    'OM', 
    'P', 
    'Q', 
    'QIS', 
    'R', 
    'S',
    'T',
    'X'
]


def run(config):
    SELECTIVE_INDEXING_PERIODS_FILEPATH = config['selective_indexing_periods_output_file']
    ENCODING = config['encoding']

    serials = _parse_serials(config)
    selective_time_periods = _extract_time_periods(serials, 'Selective')
    _serialize_time_periods(SELECTIVE_INDEXING_PERIODS_FILEPATH, ENCODING, selective_time_periods)


def _parse_serials(config):
    '''
        serials = 
        [
            {
                'nlmid': '1234', 
                'medline_ta': 'text',
                'indexing_history': [
                                        { 
                                            'date_of_action': '1978-1-1', 
                                            'citation_subset': 'H', 
                                            'indexing_treatment': 'Selective', 
                                            'indexing_status': 'Currently-indexed', 
                                            'coverage': 'text', 
                                            'coverage_note'= 'text',  
                                        },
                                    ],
            },
        ] 
    '''
    SERIALS_FILEPATH = config['serials_file']
    root_node = ET.parse(SERIALS_FILEPATH)

    serials = []  
    for serial_node in root_node.findall('Serial'):
        serial = {}
        serials.append(serial)

        nlm_unique_id_node = serial_node.find('NlmUniqueID')
        serial['nlmid'] = nlm_unique_id_node.text.strip()

        medline_ta_node = serial_node.find('MedlineTA')
        serial['medline_ta'] = medline_ta_node.text.strip() if medline_ta_node is not None else ''

        indexing_history_list = []
        indexing_history_list_node = serial_node.find('IndexingHistoryList')
        if indexing_history_list_node is not None:
            for indexing_history_node in indexing_history_list_node.findall('IndexingHistory'):
                indexing_history = {}
                indexing_history_list.append(indexing_history)

                indexing_history['citation_subset'] = indexing_history_node.attrib['CitationSubset'].strip()
                indexing_history['indexing_treatment'] = indexing_history_node.attrib['IndexingTreatment'].strip() if 'IndexingTreatment' in indexing_history_node.attrib else 'IMPLIED'
                indexing_history['indexing_status'] = indexing_history_node.attrib['IndexingStatus'].strip() if 'IndexingStatus' in indexing_history_node.attrib else 'IMPLIED'
                
                date_of_action_node = indexing_history_node.find('DateOfAction')
                year = int(date_of_action_node.find('Year').text)
                month = int(date_of_action_node.find('Month').text)
                day = int(date_of_action_node.find('Day').text)
                indexing_history['date_of_action'] = date(year, month, day)

                coverage_node = indexing_history_node.find('Coverage')
                indexing_history['coverage'] = coverage_node.text.strip().replace('\n', '') if coverage_node is not None else ''
               
                coverage_note_node = indexing_history_node.find('CoverageNote')
                indexing_history['coverage_note']= coverage_note_node.text.strip().replace('\n', '') if coverage_note_node is not None else ''

        sorted_index_history_list = sorted(indexing_history_list, key=lambda x: x['date_of_action'])
        serial['indexing_history'] = sorted_index_history_list

    return serials


def _extract_time_periods(serials, target):
    '''
    time_periods = 
        [
            {
                'nlmid': '1234',
                'medline_ta': 'text',
                'citation_subset': 'H', 
                'begin_indexing_treatment': 'Selective', 
                'begin_status': 'Currently-indexed', 
                'begin_date_of_action': '1978-1-1',
                'end_indexing_treatment': 'Selective', 
                'end_status': 'Ceased-publication', 
                'end_date_of_action': '1979-1-1',
                'coverage': 'text', 
                'coverage_note'= 'text',
                'error' = '',
            },
        ] 
    '''
    
    time_periods = []
    for serial in serials:
        for citation_subset in CITATION_SUBSETS:
            subset_indexing_history = [indexing_history for indexing_history in serial['indexing_history'] if indexing_history['citation_subset'] == citation_subset]
            expect_start = True
            for indexing_history in subset_indexing_history:
                is_target = indexing_history['indexing_treatment'] == target
                is_start = indexing_history['indexing_status'] == 'Currently-indexed'
                is_end = not is_start and indexing_history['indexing_status'] != 'IMPLIED'
                if expect_start and is_start and is_target: # Standard start
                    time_period = {
                        'nlmid': serial['nlmid'],
                        'medline_ta': serial['medline_ta'],
                        'citation_subset': indexing_history['citation_subset'], 
                        'begin_indexing_treatment': indexing_history['indexing_treatment'], 
                        'begin_status': indexing_history['indexing_status'], 
                        'begin_date_of_action': indexing_history['date_of_action'].isoformat(),
                        'coverage': indexing_history['coverage'], 
                        'coverage_note': indexing_history['coverage_note'],
                        }
                    time_periods.append(time_period)
                    expect_start = False
                elif expect_start and is_end and is_target: # Starts with end
                    time_period = {
                        'nlmid': serial['nlmid'],
                        'medline_ta': serial['medline_ta'],
                        'citation_subset': indexing_history['citation_subset'], 
                        'end_indexing_treatment' : indexing_history['indexing_treatment'],
                        'end_status' : indexing_history['indexing_status'],
                        'end_date_of_action' : indexing_history['date_of_action'].isoformat(),
                        'coverage': indexing_history['coverage'], 
                        'coverage_note': indexing_history['coverage_note'],
                        }
                    time_periods.append(time_period)
                    expect_start = True
                elif not expect_start and is_end: # Standard end
                    time_period['end_indexing_treatment'] = indexing_history['indexing_treatment']
                    time_period['end_status'] = indexing_history['indexing_status']
                    time_period['end_date_of_action'] = indexing_history['date_of_action'].isoformat()
                    expect_start = True
                elif not expect_start and not is_end:
                    time_period['error'] = 'Expected end found start'
                    break

                
    return time_periods


def _save_delimited_data(path, encoding, delimiter, data):
    with open(path, 'wt', encoding=encoding) as file:
        for data_row in data:
            line = delimiter.join([str(data_item) for data_item in data_row]) + '\n'
            file.write(line)
                    

def  _serialize_time_periods(filepath, encoding, time_periods):
    output_list = []
    headings = [
            'NLM ID',
            'MEDLINE TA',
            'CITATION SUBSET',
            'BEGIN INDEXING TREATMENT',
            'BEGIN STATUS',
            'BEGIN DATE',
            'END INDEXING TREATMENT',
            'END STATUS',
            'END DATE',
            'COVERAGE TEXT',
            'YEAR COUNT',
            'START YEAR',
            'END YEAR',
            'REVIEW COVERAGE NOTE TEXT',
            'COVERAGE NOTE TEXT',
            'REG EXP ERROR',
            'TIME PERIOD ERROR',
        ]
    output_list.append(headings)

    for time_period in time_periods:

        has_begin_status = 'begin_status' in time_period
        has_end_status = 'end_status' in time_period
        begin_status = time_period['begin_status'] if has_begin_status else ''
        end_status = time_period['end_status'] if has_end_status else ''
        
        currently_indexed = not has_end_status and has_begin_status and begin_status == 'Currently-indexed'
        
        coverage_text = time_period['coverage']
        start_year, end_year, year_count, re_error = _extract_start_end_year(coverage_text, currently_indexed)
        start_year = str(start_year)
        end_year = str(end_year)
        year_count = str(year_count)

        coverage_note_text = time_period['coverage_note']
        review_coverage_note_text = _should_review_coverage_note(coverage_note_text)

        to_output = [
            time_period['nlmid'],
            time_period['medline_ta'],
            time_period['citation_subset'],
            time_period['begin_indexing_treatment'] if has_begin_status else '',
            begin_status,
            time_period['begin_date_of_action'] if has_begin_status else '',
            time_period['end_indexing_treatment'] if has_end_status else '',
            end_status,
            time_period['end_date_of_action'] if has_end_status else '',
            coverage_text,
            year_count,
            start_year,
            end_year,
            review_coverage_note_text,
            coverage_note_text,
            re_error,
            time_period['error'] if 'error' in time_period else '',
        ]
        output_list.append(to_output)
        
    _save_delimited_data(filepath, encoding, '|', output_list)


def _extract_start_end_year(coverage_text, currently_indexed):
    start_year = -1
    end_year = -1
    re_error = ''

    matches = re.findall(r'(?:(?<=[^vn])|(?<=^))([12][09][0126789][0-9])(?=[^n]|$)', coverage_text)
    pub_years = [match for match in matches]
    
    year_count = len(pub_years)
    year_count_is_even = year_count % 2 == 0
    year_count_is_odd = not year_count_is_even

    if year_count == 0:
        re_error = 'no year in coverage text'
        return start_year, end_year, year_count, re_error

    if currently_indexed:
        if year_count_is_even:
            re_error = 'expected odd number of years for currently indexed journal'
        else:
            start_year = int(pub_years[-1])
    else:
        if year_count_is_odd:
            re_error = 'expected even number of years for previously indexed journal'
        else:
            cand_start_year = int(pub_years[-2])
            cand_end_year   = int(pub_years[-1])
            if cand_end_year >= cand_start_year:
                start_year = cand_start_year
                end_year = cand_end_year
            else:
                re_error = 'start year greater than end year'

    return start_year, end_year, year_count, re_error


def _should_review_coverage_note(coverage_note_text):
    coverage_note_text_lower = coverage_note_text.lower()
    should_review = str('sel' in coverage_note_text_lower or 'ful' in coverage_note_text_lower)
    return should_review


if __name__ == '__main__':
    config = {
        'encoding': 'utf8',
        'selective_indexing_periods_output_file' : '2022_selective_indexing_periods_output.txt',
        'serials_file': 'lsi2022_lf.xml'
    }
    run(config)