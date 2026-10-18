"""Conservative public implementation extraction, without independent certification.

Ambiguous tables, unproven restructuring and fractional allotments remain unknown.
Explicit restructuring destinations distinguish company capital from holder rights.
Extracted terms also match vendor events. Raw PDF and text hashes bind every result.
"""
import re
from decimal import Decimal
from .contracts import require, digest
from .history import iso, jq_code

DATE = r'(20\d{2})年(\d{1,2})月(\d{1,2})日'


def extract_restructuring(text, index, original, vendor_rows, source, *, allow_late=False):
    """Explicit all-new-shares-to-investors/creditors terms, never a holder split."""
    from datetime import date, timedelta
    compact=re.sub(r'\s+','',text).replace('，',',').replace('（','(').replace('）',')')
    result={'instrument':original['instrument'],'ex_date':original['ex_date'],
            'vendor_evidence_id':original['evidence_id'],'source_sha256':source['sha256']}
    def unknown(reason):return dict(result,status='unresolved',reason=reason)
    code=index['ts_code'].split('.')[0]
    header=re.search(r'证券代码[:：]?([0-9]{6})',compact[:250])
    if not header or header[1]!=code or jq_code(index['ts_code'])!=original['instrument']:
        return unknown('issuer_identity_not_matched')
    if '重整' not in compact or '资本公积转增' not in compact:
        return unknown('explicit_restructuring_plan_missing')
    record=_dates(compact,r'股权登记日')
    listing=_dates(compact,r'转增股本上市日')
    if len(record)!=1 or listing!=[original['ex_date']] or not record[0]<listing[0]:
        return unknown('record_or_listing_date_ambiguous')
    if '股权登记日次一交易日' not in compact:
        return unknown('next_session_ex_reference_missing')
    footer=re.search(r'董事会'+DATE+r'$',compact)
    ann=iso(index['ann_date'])
    if not footer or date(*(int(v) for v in footer.groups())).isoformat()!=ann:
        return unknown('publication_date_not_reproduced')
    if not allow_late and ann>record[0]:return unknown('publication_after_record_date')
    rates={Decimal(v)/10 for v in re.findall(r'每10股转增([0-9]+(?:\.[0-9]+)?)股',compact)}
    if len(rates)!=1 or next(iter(rates))<=0:return unknown('company_conversion_ratio_ambiguous')
    ratio=next(iter(rates))
    allocated=re.search(r'前述([0-9,]+)股转增股票不向原股东进行分配',compact)
    investor=re.search(r'([0-9,]+)股用于引入重整投资人',compact)
    creditor=re.search(r'([0-9,]+)股用于偿付债务',compact)
    if not (allocated and investor and creditor) or '分配给普通债权人实施以股抵债' not in compact:
        return unknown('all_new_shares_destination_not_explicit')
    number=lambda v:int(v.replace(',',''))
    total=number(allocated[1]);investor_n=number(investor[1]);creditor_n=number(creditor[1])
    if total!=investor_n+creditor_n or min(investor_n,creditor_n)<=0:
        return unknown('new_share_allocation_does_not_reconcile')
    bases={number(v) for v in re.findall(r'现有总股本([0-9,]+)股',compact)}
    if len(bases)!=1 or Decimal(next(iter(bases)))*ratio!=Decimal(total):
        return unknown('capital_base_and_conversion_do_not_reconcile')
    if '由于不涉及现金红利、股票红利及配股,公式中现金红利为0' not in compact:
        return unknown('explicit_no_cash_holder_distribution_missing')
    if any(re.search(pattern,compact) for pattern in (r'向原股东.{0,15}?(?:派发|支付)现金',r'向原股东.{0,15}?分配[0-9]+股')):
        return unknown('conflicting_holder_distribution')
    try:
        matching=[r for r in vendor_rows if jq_code(r.get('ts_code',index['ts_code']))==original['instrument']
            and iso(r.get('record_date'))==record[0] and iso(r.get('ex_date',original['ex_date']))==original['ex_date']
            and r.get('stk_div') is not None and Decimal(str(r['stk_div']))==ratio]
    except (ValueError,TypeError,ArithmeticError):return unknown('issuer_vendor_capital_event_mismatch')
    if not matching or len(matching)!=len(vendor_rows):return unknown('issuer_vendor_capital_event_mismatch')
    available=(date.fromisoformat(ann)+timedelta(days=1)).isoformat()+'T00:00:00+08:00'
    bound=dict(source,excerpt=compact[:6000],publication_index=dict(index),text_format='fitz_page_text_v1')
    judgment={'type':'public_restructuring_v1','instrument':original['instrument'],
        'ex_date':original['ex_date'],'record_date':record[0],'ann_date':ann,'pay_date':None,
        'company_share_distribution':float(ratio),'holder_cash':0.,'holder_shares':0.,'share_credit_date':None,
        'company_new_shares':total,'investor_new_shares':investor_n,'creditor_new_shares':creditor_n,
        'vendor_evidence_id':original['evidence_id'],'sources':[bound],'independent_assurance':False,
        'available_at_assumption':available,'publication_hour_verified':False,
        'availability_scope':'conservative next-calendar-day research assumption; not historical capture proof',
        'holder_scope':'ordinary preexisting stock ownership only; investor/creditor rights require separate evidence',
        'method':'explicit_restructuring_allocations_and_vendor_capital_event_v1'}
    return dict(result,status='restructuring_terms_proposed',adjudication=judgment)


def verified_restructuring_source(source):
    """Verify both archived bytes and a freshly extracted PDF text reproduction."""
    from pathlib import Path
    from .datasets import file_hash
    import fitz
    require(source['url'].startswith(('https://static.cninfo.com.cn/finalpage/',
        'https://disc.static.szse.cn/')), 'unregistered restructuring publication host')
    require(file_hash(source['path'])==source['sha256']
            and file_hash(source['text_path'])==source['text_sha256'], 'archived restructuring source changed')
    require(source.get('text_format')=='fitz_page_text_v1', 'restructuring text extraction format missing')
    with fitz.open(source['path']) as document:
        actual='\n\n'.join(page.get_text() for page in document)
    require(Path(source['text_path']).read_bytes()==actual.encode('utf8'), 'PDF and archived restructuring text differ')
    return actual


def _dates(text, label, width=40):
    from datetime import date
    found=set()
    for match in re.finditer('(?=(?:'+label+r')(?P<gap>.{0,'+str(width)+r'}?)'+DATE+')', text):
        if any(w in match.group('gap') for w in ('股权登记','除权','除息','上市日','交易日','发放日')):
            continue
        if re.search(r'[\u4e00-\u9fff]',match.group('gap').replace('为','').replace('是','')):
            continue
        try:found.add(date(*(int(x) for x in match.groups()[-3:])).isoformat())
        except ValueError:pass
    return sorted(found)


def _approved_zero_cash_transfer(compact, implemented):
    """Link explicit approved zero cash only to an unchanged same-year transfer.

    Both sections must name the same share base and transfer ratio. A missing
    implementation cash amount alone, prior-year zero, or a changed plan never
    implies zero. These remain source-bound research terms, not assurance.
    """
    if implemented==compact or any(w in implemented for w in ('现金','红利','股利','分红金额','分配金额','分红比例')) or re.search(r'每10股.{0,18}?派.{0,18}?[0-9]+(?:\.[0-9]+)?元',implemented):return False
    fiscal=re.search(r'(20\d{2})(?:年年度|年度|年)权益分派实施公告',compact[:700])
    prior=re.search(r'一、(?:股东大会|股东会)审议通过.{0,50}?方案.{0,20}?情况(?P<body>.*?)二、',compact)
    if not fiscal or not prior:return False
    approved=prior.group('body');year=fiscal.group(1)
    annual=year+r'(?:年年度|年度)(?:资本公积金转增股本方案|利润分配暨资本公积金转增股本方案|权益分派方案)'
    named=re.search(annual,approved)
    if not named or not re.search(annual,implemented):return False
    approved_plan=approved[named.start():].split('。')[0]
    if not any(w in approved_plan for w in ('不派发现金红利','不派发现金股利')):return False
    if not any(w in approved for w in ('自上述分配方案披露至本公告日,公司股本总额未发生变化',
            '自本次利润分配预案披露至本公告披露日,公司股本总额未发生变化')):return False
    rates=lambda t:{Decimal(v) for v in re.findall(r'每10股.{0,18}?转增([0-9]+(?:\.[0-9]+)?)股',t)}
    a,b=rates(approved_plan),rates(implemented)
    if len(a)!=1 or a!=b or next(iter(a))<=0:return False
    if any(re.search(r'每10股.{0,18}?送(?:红股)?[0-9]+(?:\.[0-9]+)?股',t) for t in (approved_plan,implemented)):return False
    bases=lambda t:{int(v.replace(',','')) for v in re.findall(r'总股本(?:为)?([0-9,]+)股',t)}
    a,b=bases(approved_plan),bases(implemented)
    return len(a)==1 and a==b


def extract_ordinary(text, index, original, vendor_rows, source, *, allow_missing_vendor=False):
    """Return a source-bound proposal or a named unresolved result; never infer zero.

    allow_missing_vendor admits the public_missing_vendor_event_v1 pathway: the
    vendor dividend table has no row for the ex-date, so the issuer text alone
    declares the amounts and vendor-component crosschecks are skipped by design.
    """
    # A PDF page-number line can split an explicit negative cash clause. Remove
    # only that isolated page boundary, not inline digits or financial tables.
    text=re.sub(r'(不)[ \t]*\n(?:[ \t]*\n)+[ \t]*\d{1,3}[ \t]*\n[ \t]*(派发现金(?:红利|股利))',r'\1\2',text)
    compact=re.sub(r'\s+','',text).replace('，',',').replace('（','(').replace('）',')')
    result={'instrument':original['instrument'],'ex_date':original['ex_date'],
            'vendor_evidence_id':original['evidence_id'],'source_sha256':source['sha256']}
    def unknown(reason):return dict(result,status='unresolved',reason=reason)
    if '重整' in compact or '重整' in index.get('title',''):
        return unknown('restructuring_requires_holder_specific_rights_review')
    code=index['ts_code'].split('.')[0]
    issuer_match=jq_code(index['ts_code'])==original['instrument']
    if source.get('identity_mapping_catalog_path'):
        from .datasets import file_hash
        from .contracts import load_json
        from .pit_universe import canonical_code
        path=source['identity_mapping_catalog_path']
        require(file_hash(path)==source['identity_mapping_catalog_sha256'],'issuer identity catalog changed')
        aliases=load_json(path)['identity_aliases'];stable=canonical_code(original['instrument'])
        matches=[(old,v) for old,v in aliases.items() if v['stable_code']==stable]
        require(len(matches)<=1,'ambiguous issuer historical identity')
        if matches:
            old,mapping=matches[0]
            require(mapping['mapping_evidence_id'] and iso(index['ann_date']),'issuer mapping lacks dated evidence')
            issuer_match=index['ts_code'] in (old,stable)
            code=(old if iso(index['ann_date'])<mapping['effective_session'] else stable).split('.')[0]
    if code not in compact[:700] or not issuer_match:
        return unknown('issuer_identity_not_matched')
    record=_dates(compact,r'股权登记日|权益登记日')
    ex=_dates(compact,r'除权除息日|除权\(息\)日|除权日|除息日')
    if len(record)!=1 or ex!=[original['ex_date']] or not record[0]<ex[0]:
        return unknown('record_or_ex_date_ambiguous')
    ann=iso(index['ann_date'])
    if not ann or not ann<=record[0]:return unknown('publication_after_record_date')
    amount_text=compact
    headings=('一、权益分派方案','二、权益分派方案','二、本次权益分派方案','二、本次实施的利润分配及资本公积转增股本方案',
        '二、本次实施的权益分派方案','二、本次实施的转增股本方案','二、本次实施的分配方案',
        '二、本次实施的资本公积金转增股本方案',
        '二、分配方案','二、权益分配方案','三、权益分派方案')
    spans=[]
    for heading in headings:
        start=compact.find(heading)
        if start<0:continue
        start+=len(heading);end=re.search(r'[二三四五六七]、',compact[start:])
        spans.append(compact[start:start+end.start()] if end else compact[start:])
    if len(set(spans))==1:amount_text=spans[0]
    # NEEQ/BSE implementation templates distinguish the gross plan from the
    # separately labelled withholding section. Never parse after-tax rates.
    gross_plan=amount_text.split('2、扣税说明')[0] if '2、扣税说明' in amount_text else amount_text
    zero_cash=any(w in amount_text for w in ('不进行现金分红','不派发现金红利','不派发现金股利','现金分红为0','不进行现金红利分配'))
    if not zero_cash and any(w in compact for w in ('本次实施的分派方案与股东大会审议通过的分配预案及其调整原则是一致的',
            '与股东大会审议通过的方案及其调整原则一致')):
        explicit_zero=any(w in compact for w in ('不派发现金红利','不派发现金股利'))
        positive_cash=re.search(r'每10股.{0,12}?派.{0,12}?[1-9][0-9]*(?:\.[0-9]+)?元',compact)
        zero_cash=explicit_zero and not positive_cash
    linked_approved_zero=not zero_cash and _approved_zero_cash_transfer(compact,amount_text)
    zero_cash=zero_cash or linked_approved_zero
    cash=set()
    for m in re.finditer(r'每10股.{0,14}?派(?:发|送)?(?:人民币)?(?:现金红利|现金股利|现金)?(?:人民币)?([0-9]+(?:\.[0-9]+)?)元.{0,16}?含税',amount_text):
        cash.add(Decimal(m.group(1))/10)
    if '2、扣税说明' in amount_text and '差别化个人所得税' in amount_text:
        for m in re.finditer(r'每10股派人民币现金([0-9]+(?:\.[0-9]+)?)元',gross_plan):
            cash.add(Decimal(m.group(1))/10)
        for m in re.finditer(r'每10股派([0-9]+(?:\.[0-9]+)?)元人民币现金',gross_plan):
            cash.add(Decimal(m.group(1))/10)
    if zero_cash:
        if cash and cash!={Decimal(0)}:return unknown('conflicting_public_cash_rates')
        cash={Decimal(0)}
    if len(cash)!=1:return unknown('explicit_gross_cash_rate_missing_or_ambiguous')
    share_plan=gross_plan
    # A split by taxable reserve type explains one total; verify its sum before
    # removing the explanation so 2 total + 2 exempt + 0 taxable is not 4 shares.
    for m in list(re.finditer(r'\(其中以股票发行溢价形成的资本公积金.*?\)',share_plan)):
        rates=[Decimal(v) for v in re.findall(r'每10股转增([0-9]+(?:\.[0-9]+)?)股',m.group())]
        preceding=re.search(r'每10股转增([0-9]+(?:\.[0-9]+)?)股[,。]?$',share_plan[:m.start()])
        if preceding and len(rates)==2 and sum(rates,Decimal(0))==Decimal(preceding.group(1)):
            share_plan=share_plan.replace(m.group(),'')
    transfer={Decimal(m.group(1))/10 for m in re.finditer(r'每10股.{0,18}?转增([0-9]+(?:\.[0-9]+)?)股',share_plan)}
    bonus={Decimal(m.group(1))/10 for m in re.finditer(r'每10股.{0,18}?送(?:红股)?([0-9]+(?:\.[0-9]+)?)股',share_plan)}
    if len(transfer)>1 or len(bonus)>1:return unknown('aggregate_and_holder_share_rates_ambiguous')
    try:
        observed_shares={Decimal(str(r['stk_div'])) for r in vendor_rows if r.get('stk_div') is not None}
        observed_cash={Decimal(str(r['cash_div_tax'])) for r in vendor_rows if r.get('cash_div_tax') is not None}
    except Exception:return unknown('vendor_amount_not_numeric')
    missing_vendor=bool(allow_missing_vendor) and not vendor_rows
    require(not missing_vendor or not (observed_shares or observed_cash),
            'missing-vendor extraction requires empty vendor rows')
    shares=sum(transfer,Decimal(0))+sum(bonus,Decimal(0))
    if not transfer and not bonus:
        if observed_shares!={Decimal(0)} and not missing_vendor:return unknown('explicit_holder_share_rate_missing')
        if ('转增' in share_plan and not any(w in share_plan for w in ('不转增','不进行资本公积金转增','不进行公积金转增','不以资本公积金转增','不以公积金转增'))) or ('送红股' in share_plan and '不送红股' not in share_plan):
            return unknown('unparsed_public_share_terms')
    gross=next(iter(cash))
    aggregate=(shares==0 and observed_shares=={Decimal(0)} and gross>0 and '合计' in amount_text and
               gross==sum(observed_cash,Decimal(0)) and all(v>0 for v in observed_cash) and
               all(iso(r.get('record_date'))==record[0] for r in vendor_rows))
    if not missing_vendor and (shares not in observed_shares or (gross>0 and gross not in observed_cash and not aggregate)):
        return unknown('public_amounts_do_not_match_vendor_component')
    if not missing_vendor and not any(iso(r.get('record_date'))==record[0] and
               r.get('stk_div') is not None and
               Decimal(str(r.get('stk_div')))==shares and
               (gross==0 or aggregate or r.get('cash_div_tax') is not None and Decimal(str(r['cash_div_tax']))==gross)
               for r in vendor_rows):return unknown('no_matching_vendor_entitlement_component')
    pay=None;credit=None
    if gross>0:
        pays=_dates(compact,r'现金红利发放日|现金红利派发日|现金红利到账日|现金红利(?:将)?于')
        if len(pays)!=1 or pays[0]<ex[0]:return unknown('cash_payment_date_missing_or_ambiguous')
        pay=pays[0]
    if shares>0:
        credits=_dates(compact,r'本次(?:所)?(?:转增(?:的)?(?:股份)?|转股|送\(转\)股)(?:将)?于|本次送转股份于|本公司此次权益分派转增的股份将于|本次权益分派所转股将于',width=8)
        tradable=_dates(compact,r'起始交易日(?:为)?|新增无限售条件流通股份(?:的)?上市日(?:为)?|新增可流通股份上市日(?:为)?',width=8)
        if len(credits)!=1 or tradable!=credits or credits[0]<ex[0]:
            return unknown('separate_credit_and_tradability_dates_not_matched')
        credit=credits[0]
    source=dict(source,excerpt=compact[:4500],publication_index=dict(index))
    # Buyback-adjusted implementations declare the exchange ex-rights deduction
    # per total share, which differs from the per-holder cash rate. It is the
    # figure the exchange enforces in the pre_close chain on the ex-date.
    reference_deduction=None
    refs={Decimal(m.group(1)) for m in re.finditer(r'除权除息参考价格.{0,60}?([0-9]+\.[0-9]+)元/股',compact)}
    if len(refs)>1:return unknown('ambiguous_exchange_reference_deduction')
    if refs:reference_deduction=float(next(iter(refs)))
    judgment={'type':'public_implementation_v1','instrument':original['instrument'],
        'ex_date':original['ex_date'],'record_date':record[0],'ann_date':ann,'pay_date':pay,
        'holder_cash':float(gross),'holder_shares':float(shares),'share_credit_date':credit,
        'vendor_evidence_id':original['evidence_id'],'sources':[source],
        'independent_assurance':False,'fractional_allocation':'unknown; integral entitlements only',
        'method':'conservative_text_and_vendor_component_crosscheck_v1'}
    if reference_deduction is not None:judgment['reference_cash_deduction']=reference_deduction
    if linked_approved_zero:judgment['cash_zero_basis']='explicit approved same-year zero cash; implemented transfer ratio and unchanged share base matched'
    if aggregate:judgment['vendor_cash_match']='public explicit aggregate equals distinct vendor rates; not sum of unlabelled revisions'
    return dict(result,status='ordinary_terms_proposed',adjudication=judgment)
