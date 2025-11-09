import pandas as pd
from datasets_.util import standardize_bcp47
from pathlib import Path
import tarfile
import requests

fleurs_tags = "af_za,am_et,ar_eg,as_in,ast_es,az_az,be_by,bg_bg,bn_in,bs_ba,ca_es,ceb_ph,ckb_iq,cmn_hans_cn,cs_cz,cy_gb,da_dk,de_de,el_gr,en_us,es_419,et_ee,fa_ir,ff_sn,fi_fi,fil_ph,fr_fr,ga_ie,gl_es,gu_in,ha_ng,he_il,hi_in,hr_hr,hu_hu,hy_am,id_id,ig_ng,is_is,it_it,ja_jp,jv_id,ka_ge,kam_ke,kea_cv,kk_kz,km_kh,kn_in,ko_kr,ky_kg,lb_lu,lg_ug,ln_cd,lo_la,lt_lt,luo_ke,lv_lv,mi_nz,mk_mk,ml_in,mn_mn,mr_in,ms_my,mt_mt,my_mm,nb_no,ne_np,nl_nl,nso_za,ny_mw,oc_fr,om_et,or_in,pa_in,pl_pl,ps_af,pt_br,ro_ro,ru_ru,sd_in,sk_sk,sl_si,sn_zw,so_so,sr_rs,sv_se,sw_ke,ta_in,te_in,tg_tj,th_th,tr_tr,uk_ua,umb_ao,ur_pk,uz_uz,vi_vn,wo_sn,xh_za,yo_ng,yue_hant_hk,zu_za"

fleurs = pd.DataFrame(fleurs_tags.split(","), columns=["fleurs_tag"])
fleurs["bcp_47"] = fleurs["fleurs_tag"].apply(
    lambda x: standardize_bcp47(x.rsplit("_")[0], macro=True)
)


def download_file(url, path):
    response = requests.get(url)
    with open(path, "wb") as f:
        f.write(response.content)


def download_fleurs(transcription_langs_eval):
    # the huggingface loader does not allow loading only the dev set, so do it manually
    for language in transcription_langs_eval.itertuples():
        tar_url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{language.fleurs_tag}/audio/dev.tar.gz"
        tar_path = Path(f"data/fleurs/{language.fleurs_tag}/audio/dev.tar.gz")
        audio_path = Path(f"data/fleurs/{language.fleurs_tag}/audio")
        if not audio_path.exists():
            print(f"Downloading {tar_url} to {tar_path}")
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            download_file(tar_url, tar_path)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=audio_path)
        tsv_url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{language.fleurs_tag}/dev.tsv"
        tsv_path = Path(f"data/fleurs/{language.fleurs_tag}/dev.tsv")
        if not tsv_path.exists():
            print(f"Downloading {tsv_url} to {tsv_path}")
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            download_file(tsv_url, tsv_path)
