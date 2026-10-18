"""Keep every sealed test/source receipt; create final rc10 variants."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for old,new in [('fuxi_notice_tests_v3_20261003.py','fuxi_notice_tests_v4_20261003.py'),
                ('fuxi_notice_adjudicate_v2_20261003.py','fuxi_notice_adjudicate_v3_20261003.py')]:
    content=(ROOT/'scripts'/old).read_bytes().decode('utf-8')
    if 'tests_' in old:content=content.replace('tests_v3','tests_v4')
    else:content=content.replace('PUBLIC_IMPLEMENTATION_BUNDLE_v2.json','PUBLIC_IMPLEMENTATION_BUNDLE_v3.json')
    with (ROOT/'scripts'/new).open('xb') as handle:handle.write(content.encode('utf-8'))
path=ROOT/'scripts/fuxi_notice_ten_rounds_v3_20261003.py'
s=path.read_bytes().decode('utf-8').replace('tests_v3','tests_v4').replace('PUBLIC_IMPLEMENTATION_BUNDLE_v2.json','PUBLIC_IMPLEMENTATION_BUNDLE_v3.json')
path.write_bytes(s.encode('utf-8'))
print('final rc10 variants written; old successful tests and source bundles retained')
