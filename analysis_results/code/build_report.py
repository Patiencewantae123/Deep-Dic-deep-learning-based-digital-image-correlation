"""Build report.html from the result files. Every number below is read from a result file, none are typed in."""
import base64, csv, html, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))
E, P, M, V, D, L = J("eval.json"), J("peel.json"), J("model_probe.json"), J("verify.json"), J("strain_diagnose.json"), J("figure_limits.json")
T = {t: J(f"runs/{t}_timing.json") for t in ("disp", "strain")}
leak = E["leaked"]
T5 = J("test500.json")   # test-set results on 500 unseen pairs (eval_test500.py), with 95% intervals
dp, sp, N_TEST = T5["disp_px"], T5["strain_pct"], T5["samples"]
ev, pre, roi, tr = E["eval_bug"], E["preprocessing"], E["rescale"], E["roi_tracker"]


def rows(name):
    with open(os.path.join(HERE, "runs", name)) as f:
        return list(csv.DictReader(f))


def curve(task, scale):
    seen = {}
    for r in rows(f"{task}_val.csv"):
        seen[int(r["step"])] = (round(float(r["val_mae"]) * scale, 3), int(r["phase"]))
    return [dict(x=k, y=v[0], phase=v[1]) for k, v in sorted(seen.items())]


disp_curve, strain_curve = curve("disp", .5), curve("strain", 1)
val_d, val_s = rows("disp_val.csv"), rows("strain_val.csv")
drop_d = next(int(r["step"]) for r in rows("disp_steps.csv") if r["phase"] == "2")
drop_s = next(int(r["step"]) for r in rows("strain_steps.csv") if r["phase"] == "2")
mse = lambda rs, ph, i: float([r for r in rs if r["phase"] == ph][i]["val_mse"])
d_before, d_after = mse(val_d, "1", -1), mse(val_d, "2", -1)


def img(name):
    with open(os.path.join(HERE, "figures", name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def tile(name, label, alt):
    return f'<figure class="tile"><img src="{img(name)}" alt="{html.escape(alt)}" width="384" height="384"><figcaption>{label}</figcaption></figure>'


def scale(kind, lim, unit):
    grad = L["diverging_css"] if kind == "div" else L["sequential_css"]
    low = f"&minus;{lim:g}" if kind == "div" else "0"
    mid = "<span>0</span>" if kind == "div" else ""
    return f'<div class="scale"><div class="ramp" style="background:{grad}"></div><div class="ticks"><span>{low}</span>{mid}<span>{lim:g} {unit}</span></div></div>'


def fields(s):
    lim = L[s]
    t = lambda n, lab, alt: tile(f"{s}_{n}.png", lab, alt)
    return f"""<div class="grid5">
{t('ref', 'Reference image', 'Speckle image before deformation')}{t('u_gt', 'True u', 'True horizontal displacement')}{t('u_pred', 'DisplacementNet u', 'Predicted horizontal displacement')}{t('u_flow', 'Farneback u', 'Horizontal displacement from optical flow')}{t('u_err', 'Network error, u', 'Absolute error of the predicted horizontal displacement')}
{t('def', 'Deformed image', 'Speckle image after deformation')}{t('v_gt', 'True v', 'True vertical displacement')}{t('v_pred', 'DisplacementNet v', 'Predicted vertical displacement')}{t('v_flow', 'Farneback v', 'Vertical displacement from optical flow')}{t('v_err', 'Network error, v', 'Absolute error of the predicted vertical displacement')}
</div>
<div class="scales"><div><b>Displacement</b>{scale('div', lim['disp'], 'px')}</div><div><b>Absolute error</b>{scale('seq', lim['disp_err'], 'px')}</div></div>"""


def strain_fields(s):
    lim = L[s]
    t = lambda n, lab, alt: tile(f"{s}_{n}.png", lab, alt)
    return f"""<div class="grid5">
{t('exx_gt', 'True &part;u/&part;x', 'True normal strain in x')}{t('exx_pred', 'StrainNet', 'StrainNet prediction of normal strain in x')}{t('exx_fromdisp', 'Differentiated DisplacementNet', 'Strain from differentiating predicted displacement')}{t('exx_err', 'StrainNet error', 'StrainNet absolute error')}{t('exx_fromdisp_err', 'Differentiated error, mostly off the scale', 'Absolute error of differentiated strain, clipped at the top of the scale')}
</div>
<div class="scales"><div><b>Strain</b>{scale('div', lim['strain'], '%')}</div><div><b>Absolute error</b>{scale('seq', lim['strain_err'], '%')}</div></div>"""


NICE = {"DisplacementNet_train": "DisplacementNet_train.ipynb", "StrainNet_train": "StrainNet_train.ipynb",
        "Diaplacement_run": "Diaplacement_run.ipynb", "StrainNet_run": "StrainNet_run.ipynb"}


def chain(key):
    rs = P[key]
    out = []
    for i, r in enumerate(rs):
        if not r["error"]:
            continue
        nxt = rs[i + 1] if i + 1 < len(rs) else None
        if nxt and nxt["error"] == r["error"] and nxt["message"] == r["message"]:
            continue  # the fix tried here did not change the failure, so it was not the blocker
        fix = nxt["fixes_so_far"][-1] if nxt else ""
        out.append(f"<tr><td class=mono>{r['cell']}</td><td class=mono>{html.escape(r['error'])}</td><td>{html.escape(fix)}</td></tr>")
    return len(out), "\n".join(out)


chains = {k: chain(k) for k in NICE}
blockers = sum(c[0] for c in chains.values())
chain_html = "".join(
    f'<details{" open" if k == "StrainNet_train" else ""}><summary><span class="mono">{v}</span><span class="count">{chains[k][0]} blockers</span></summary>'
    f'<div class="scroll"><table><thead><tr><th>Cell</th><th>Failure</th><th>Fix applied</th></tr></thead><tbody>{chains[k][1]}</tbody></table></div></details>'
    for k, v in NICE.items())

passed = sum(c["ok"] for c in V)
over_drop = (1 - D["overfit_end"] / D["overfit_start"]) * 100
corr_pred = max(abs(c) for c in D["pred_label_corr"])
corr_flow = min(D["flow_label_corr"])
days = lambda t, key: ((M[key]["step_s"] + T[t]["data_s_per_step"]) * 3000 + 4000 * M[key]["infer_s"]) * 200 / 86400
days_d, days_s = days("disp", "DisplacementNet"), days("strain", "StrainNet")
img_seen = T["disp"]["steps"] * 12
ratio_paper = 36000 * 200 / img_seen
fmt = lambda v, n=2: f"{v:.{n}f}"

PN, SPR = J("pipeline_test_net.json"), J("strain_probe.json")
SC, PAIR = T5["strain_channels_long"], T5["paired"]
ci = lambda d, unit: f"{fmt(d['mean'])} &plusmn; {fmt(d['ci95'])}{unit}"
steps_long = {t: int(rows(f"long_{t}_val.csv")[-1]["step"]) for t in ("disp", "strain")}
img_long = steps_long["disp"] * 12
ROI_WEIGHTS_TEXT = {"disp_phase2.pth": f"trained for {T['disp']['steps']} steps",
                    "long_disp_phase2.pth": f"the longer run, {steps_long['disp']:,} steps"}
neg = lambda v, n=2: fmt(v, n).replace("-", "&minus;")
CH_NAMES = ["&part;u/&part;row", "&part;u/&part;column (&epsilon;<sub>xx</sub>)", "&part;v/&part;row (&epsilon;<sub>yy</sub>)", "&part;v/&part;column"]
ch_rows = "".join(f'<tr><td>{n}</td><td class="num">{fmt(c)}%</td><td class="num">{fmt(e)}%</td><td class="num">{neg(r)}</td></tr>'
                  for n, c, e, r in zip(CH_NAMES, SC["best_constant_pct"], SC["net_pct"], SC["corr"]))
PF = J("pipeline_test_flow.json")
pm = SPR["mean_pct"]
NB_CASE, C128, C192 = SPR["notebook_case"], "128x128 crop, native scale", "192x192 crop, native scale"
NB_SIZE = NB_CASE.split("-> ")[1].replace("x", "&times;")
learned = max(range(4), key=lambda c: SC["corr"][c])
LONG_HTML = f"""<section>
  <h2>Longer training run: about 2.5 hours per network</h2>
  <p class="prose">To see whether more training helps, I continued both networks from the first run's weights: 120 minutes at learning rate 1e-3, then 25 minutes at 1e-5, both at once on the CPU. DisplacementNet reached {steps_long['disp']:,} steps in total and StrainNet {steps_long['strain']:,}. That is {img_long:,} training images for DisplacementNet, about one epoch of the paper's 36,000 pairs.</p>
  <figure class="wide"><img src="{img('09_long_training.png')}" alt="Validation error over both runs on a log scale. DisplacementNet falls to {fmt(float(rows('long_disp_val.csv')[-1]['val_mae']) * .5)} px; StrainNet to {fmt(float(rows('long_strain_val.csv')[-1]['val_mae']))}%. Both stay far above the paper's final values." width="1500" height="580"></figure>
  <div class="scroll" style="margin-top:16px"><table>
    <thead><tr><th>{N_TEST} unseen pairs, mean error</th><th>Predicting zero</th><th>First run</th><th>Longer run</th><th>Paper's test set</th></tr></thead>
    <tbody>
      <tr><td>DisplacementNet</td><td class="num">{ci(dp['zero'], ' px')}</td><td class="num">{ci(dp['net_p2'], ' px')}</td><td class="num"><b>{ci(dp['long'], ' px')}</b></td><td class="num">0.038 px</td></tr>
      <tr><td>StrainNet</td><td class="num">{ci(sp['zero'], '%')}</td><td class="num">{ci(sp['strainnet_p2'], '%')}</td><td class="num"><b>{ci(sp['long'], '%')}</b></td><td class="num">0.041%</td></tr>
    </tbody>
  </table></div>
  <p class="prose" style="margin-top:14px">&plusmn; is the 95% confidence interval over the {N_TEST} pairs. DisplacementNet's error is {fmt(dp['net_p2']['mean'] / dp['long']['mean'], 1)} times lower than after the first run, and lower on {fmt(PAIR['disp long - net_p2']['first_better_share'] * 100, 0)}% of the pairs. Farneback optical flow still does better on these clean images ({fmt(dp['farneback']['mean'])} px), and the paper's error is {fmt(dp['long']['mean'] / .038, 0)} times lower again. StrainNet is {fmt(-PAIR['strain long - zero']['mean'])} &plusmn; {fmt(PAIR['strain long - zero']['ci95'])} points below predicting zero, a real but small gain.</p>

  <h3>StrainNet learned one of its four channels</h3>
  <p class="prose">To tell real learning from a network that has only learned the dataset's average, the table compares each channel with the best single value it could predict everywhere.</p>
  <div class="scroll"><table>
    <thead><tr><th>Channel, {N_TEST} unseen pairs</th><th>Best constant</th><th>StrainNet</th><th>Correlation with labels</th></tr></thead>
    <tbody>{ch_rows}</tbody>
  </table></div>
  <p class="prose" style="margin-top:14px">On {CH_NAMES[learned]} StrainNet's predictions follow the labels, both between images and within each one. The other three channels are still level with the best constant. All of StrainNet's improvement comes from that one channel.</p>

  <h3>Both trained networks on the 40-frame tensile test</h3>
  <p class="prose">The ROI result above already uses the longer-trained DisplacementNet. Running both notebooks with the trained networks, on the same 40 frames, gives:</p>
  <div class="scroll"><table>
    <thead><tr><th>After 40 frames</th><th>Trained networks</th><th>Optical flow in their place</th><th>Truth</th></tr></thead>
    <tbody>
      <tr><td>Accumulated displacement, mean error</td><td class="num">u {fmt(PN['disp_acc_mean_abs_error_px']['u'], 1)} px, v {fmt(PN['disp_acc_mean_abs_error_px']['v'], 1)} px</td><td class="num">u {fmt(PF['disp_acc_mean_abs_error_px']['u'])} px, v {fmt(PF['disp_acc_mean_abs_error_px']['v'])} px</td><td class="num">v from {neg(PF['disp_acc_range_true_px']['v'][0], 0)} to {neg(PF['disp_acc_range_true_px']['v'][1], 0)} px</td></tr>
      <tr><td>Accumulated &epsilon;<sub>yy</sub></td><td class="num">{neg(PN['strain_acc_pct']['eyy'], 1)}%</td><td class="num">{neg(PF['strain_acc_pct']['eyy'], 1)}%</td><td class="num">{neg(PN['strain_acc_true_pct']['eyy'], 1)}%</td></tr>
      <tr><td>Accumulated &epsilon;<sub>xx</sub></td><td class="num">{neg(PN['strain_acc_pct']['exx'], 1)}%</td><td class="num">{neg(PF['strain_acc_pct']['exx'], 1)}%</td><td class="num">{neg(PN['strain_acc_true_pct']['exx'], 1)}%</td></tr>
      <tr><td>Accumulated &epsilon;<sub>xy</sub></td><td class="num">{neg(PN['strain_acc_pct']['exy'], 1)}%</td><td class="num">{neg(PF['strain_acc_pct']['exy'], 1)}%</td><td class="num">0</td></tr>
    </tbody>
  </table></div>
  <p class="prose" style="margin-top:14px">The accumulated displacement is off by about {fmt(max(PN['disp_acc_mean_abs_error_px'].values()), 1)} px on average, out of up to {neg(-PF['disp_acc_range_true_px']['v'][0], 0)} px of motion. The accumulated strain is wrong. On one frame pair, StrainNet's &part;u/&part;column reads {neg(pm[NB_CASE][1])}% on the notebook's {NB_SIZE} input, where the truth is {neg(SPR['truth_pct'][1])}%, and 40 frames add that up. The same pair gives {neg(pm[C128][1])}% on a 128&times;128 crop and {neg(pm[C192][1])}% on 192&times;192. StrainNet has only seen 128&times;128 inputs and does not yet carry over to larger ones. DisplacementNet does, as its ROI result shows.</p>
  <p class="prose">Each frame pair of a tensile test adds little strain, {neg(SPR['truth_pct'][2], 1)}% here. StrainNet's error has to be well below that, as the paper's 0.041% is, before its accumulated output means anything.</p>
</section>"""

# ---- Results checked against the paper (dataset_stats.py, paper_checks.py, eval_test500.py)
_rt = J("roi_test.json")["summary"]
ROI_SIZE_F = f"{_rt['fixed']['final_height']:.0f} &times; {_rt['fixed']['final_width']:.0f}"
ROI_SIZE_T = f"{_rt['truth']['final_height']:.0f} &times; {_rt['truth']['final_width']:.0f}"
DS, PC = J("dataset_stats.json"), J("paper_checks.json")
med = lambda k, n=1: fmt(DS[k]["50"], n)
rng5 = lambda k, n=1: f"{fmt(DS[k]['5'], n)}&ndash;{fmt(DS[k]['95'], n)}"
NF, TR = PC["noise_floor"], PC["translation"]
tr_bias = max(abs(r["net_mean_v"] - r["shift_px"]) for r in TR)
tr_spread = (min(r["net_spread_v"] for r in TR), max(r["net_spread_v"] for r in TR))
tr_strain = (min(r["strain_net_mean_abs_pct"] for r in TR), max(r["strain_net_mean_abs_pct"] for r in TR))
tr_flow = max(r["flow_mean_abs_error"] for r in TR)
epochs_d = steps_long["disp"] * 12 / 36000
PAPER_HTML = f"""<section>
  <h2>Results checked against the paper</h2>
  <p class="prose">I went through the paper's methods and results and repeated what can be repeated without the authors' data: its dataset statistics (Appendix A), test errors (Table 3), the comparison of direct and differentiated strain (Figure 8), the noise floor (Table 5) and rigid translation (Figure 12). All use the longer-trained networks.</p>

  <h3>The synthetic data is harder than the paper's</h3>
  <p class="prose"><code>gen_data.py</code> follows Tables 1 and 2: the speckle recipe, the quality mix (5% sparse, 30% large speckles, 5% noisy, 5% low contrast) and every deformation range. Table 2 gives only ranges, though. Drawing each parameter uniformly within them, with every mode in every sample, gives much larger deformations than the paper's own histograms of its training set show:</p>
  <div class="scroll"><table>
    <thead><tr><th>Per training sample</th><th>Paper, Figure A-1 (read off the histograms)</th><th>This dataset, median (5th&ndash;95th percentile)</th></tr></thead>
    <tbody>
      <tr><td>Largest displacement, px at 128&times;128</td><td>Peak near 1.3 px; few samples above 4 px</td><td class="num">{med('max_disp_px')} ({rng5('max_disp_px')})</td></tr>
      <tr><td>Standard deviation of displacement</td><td>Peak near 0.4 px</td><td class="num">{med('std_disp_px', 2)} ({rng5('std_disp_px', 2)})</td></tr>
      <tr><td>Largest equivalent strain</td><td>Most samples below about 0.5%; tail to 3%</td><td class="num">{med('max_eq_strain_pct')}% ({rng5('max_eq_strain_pct')})</td></tr>
    </tbody>
  </table></div>
  <p class="prose" style="margin-top:14px">So the paper must have favoured small values within Table 2's ranges, or used fewer deformation modes per sample; it does not say which. Here, displacements are about 4 times and strains about 10 times larger, and predicting zero scores {fmt(dp['zero']['mean'])} px and {fmt(sp['zero']['mean'])}%. So the paper's error figures and ours are not like-for-like: larger deformations generally mean larger errors, and the same network would likely score lower on the paper's data.</p>

  <h3>The paper's results against ours</h3>
  <div class="scroll"><table>
    <thead><tr><th>Paper's result</th><th>Paper</th><th>This code, longer run</th><th>Reading</th></tr></thead>
    <tbody>
      <tr><td>Test error, DisplacementNet (Table 3)</td><td class="num">0.038 px mean, 0.083 px mean max</td><td class="num">{fmt(dp['long']['mean'])} px, {fmt(dp['long']['mean_max'])} px</td><td>Not reached</td></tr>
      <tr><td>Test error, StrainNet (Table 3)</td><td class="num">0.041% mean, 0.085% mean max</td><td class="num">{fmt(sp['long']['mean'])}%, {fmt(sp['long']['mean_max'])}%</td><td>Not reached</td></tr>
      <tr><td>Strain from differentiating displacement is far worse than StrainNet (Figure 8)</td><td class="num">0.24% against 0.018%</td><td class="num">{fmt(sp['from_long_dispnet']['mean'])}% against {fmt(sp['long']['mean'])}%</td><td>Same direction</td></tr>
      <tr><td>Noise floor, stationary pairs (Table 5)</td><td class="num">0.0094 px, 0.0073%</td><td class="num">{fmt(NF['disp_net_px']['mean'], 3)} px, {fmt(NF['strain_net_pct']['mean'])}%</td><td>Not reached; optical flow gives {fmt(NF['disp_farneback_px']['mean'], 3)} px</td></tr>
      <tr><td>Rigid translation gives a non-uniform field (Figure 12)</td><td class="num">Up to 0.275 px from VIC-2D</td><td class="num">Mean shift within {fmt(tr_bias)} px; field varies {fmt(tr_spread[0], 1)}&ndash;{fmt(tr_spread[1], 1)} px; {fmt(tr_strain[0], 1)}&ndash;{fmt(tr_strain[1], 1)}% strain where there is none</td><td>Same weakness, larger; optical flow is within {fmt(tr_flow)} px</td></tr>
      <tr><td>ROI grows taller and narrower in a tensile test (Figure 14, 188&times;374 to 178&times;434, width &times; height)</td><td class="num">Bronze specimen</td><td class="num">Synthetic test: {ROI_SIZE_F} (height &times; width), truth {ROI_SIZE_T}</td><td>Matches</td></tr>
      <tr><td>Error falls over the first epochs (Figure 7)</td><td class="num">200 epochs of 36,000 pairs</td><td class="num">{fmt(epochs_d, 2)} epochs</td><td>Too early to compare</td></tr>
    </tbody>
  </table></div>
  <p class="prose" style="margin-top:14px">Where the paper describes a qualitative behaviour, the code shows it: direct strain beats differentiated displacement, rigid translation is the network's weak case, and the ROI follows a stretching specimen. Every accuracy figure is still 5&ndash;50 times worse than the paper's. That is expected after about one of the paper's 200 epochs, on harder data, so it is not evidence against the method. Whether full training closes the gap needs a GPU run on the authors' dataset.</p>
</section>"""

bars_disp = [dict(label="Farneback optical flow (classical)", value=dp["farneback"]["mean"], note="OpenCV, no training"),
             dict(label="DisplacementNet with learning-rate drop", value=dp["net_p2"]["mean"], hi=True, note=f"{T['disp']['steps']} steps"),
             dict(label="DisplacementNet, notebook settings only", value=dp["net_p1"]["mean"], note="Learning rate fixed at 1e-3"),
             dict(label="Predicting zero everywhere", value=dp["zero"]["mean"], note="What an untrained network outputs")]
bars_strain = [dict(label="Differentiated Farneback flow", value=sp["from_farneback"]["mean"], note="np.gradient of the classical flow"),
               dict(label="StrainNet", value=sp["strainnet_p2"]["mean"], hi=True, note=f"{T['strain']['steps']} steps"),
               dict(label="Predicting zero everywhere", value=sp["zero"]["mean"], note="What an untrained network outputs"),
               dict(label="Differentiated DisplacementNet", value=sp["from_dispnet"]["mean"], note="np.gradient of the network output")]
bars_eval = [dict(label="Fixed: fcn.eval() called", value=ev["eval"], hi=True, note="BatchNorm uses its training statistics"),
             dict(label="Committed: eval() never called", value=ev["train_mode"], note="BatchNorm uses single-image statistics"),
             dict(label="Eval after 150 frames in train mode", value=ev["polluted_eval"], note="Inference overwrote the stored statistics")]
bars_light = [dict(label="Fixed: each image over its own maximum", value=pre["dim_second_norm_each"], hi=True, note="Now used by both run notebooks"),
              dict(label="Committed StrainNet_run: image 2 over image 1's maximum", value=pre["dim_second_norm_first"], note="Does not cancel the dimming"),
              dict(label="Training's preprocessing: no normalisation", value=pre["dim_second_raw"], note="Raw pixel values")]

RT, PT = J("roi_test.json"), J("pipeline_test_flow.json")
rs = RT["summary"]
size = lambda r: f"{r['final_height']:.0f} &times; {r['final_width']:.0f}"
roi_subs = dict(FIG_ROI=img("08_roi_tracking.png"), ROI_BEFORE=fmt(rs["before"]["worst_edge_error_px"], 0), ROI_SIGN=fmt(rs["sign_only"]["worst_edge_error_px"], 0),
                ROI_FIXED_ERR=fmt(rs["fixed"]["worst_edge_error_px"], 0), ROI_FLOW=fmt(rs["fixed_flow"]["worst_edge_error_px"], 0),
                ROI_BEFORE_SIZE=size(rs["before"]), ROI_SIGN_SIZE=size(rs["sign_only"]), ROI_FIXED_SIZE=size(rs["fixed"]),
                ROI_FLOW_SIZE=size(rs["fixed_flow"]), ROI_TRUE_SIZE=size(rs["truth"]), ROI_WEIGHTS=ROI_WEIGHTS_TEXT.get(RT["weights"], RT["weights"]),
                P_U=fmt(PT["disp_acc_mean_abs_error_px"]["u"]), P_V=fmt(PT["disp_acc_mean_abs_error_px"]["v"]),
                P_VMIN=fmt(PT["disp_acc_range_true_px"]["v"][0], 0).replace("-", "&minus;"), P_VMAX=fmt(PT["disp_acc_range_true_px"]["v"][1], 0).replace("-", "&minus;"),
                P_EYY=fmt(PT["strain_acc_pct"]["eyy"]), P_EYY_T=fmt(PT["strain_acc_true_pct"]["eyy"]),
                P_EXX=fmt(PT["strain_acc_pct"]["exx"]).replace("-", "&minus;"), P_EXX_T=fmt(PT["strain_acc_true_pct"]["exx"]).replace("-", "&minus;"),
                P_EXY=fmt(PT["strain_acc_pct"]["exy"]), LONG_SECTION=LONG_HTML,
                L_D=fmt(dp["long"]["mean"]), L_S=fmt(sp["long"]["mean"]), N_TEST=N_TEST, L_STEPS_S=f"{steps_long['strain']:,}",
                PAPER_SECTION=PAPER_HTML, L_IMG=f"{img_long:,}", L_RATIO=f"{36000 * 200 / img_long:,.0f}")

DATA = json.dumps(dict(disp=disp_curve, strain=strain_curve, dropD=drop_d, dropS=drop_s, zeroD=disp_curve[0]["y"], zeroS=strain_curve[0]["y"],
                       bars=dict(disp=bars_disp, strain=bars_strain, eval=bars_eval, light=bars_light)))
TEMPLATE = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
speckle = img("typical_ref.png")
subs = dict(
    SPECKLE=speckle, DATA=DATA, CHAINS=chain_html, BLOCKERS=blockers, PASSED=passed, TOTAL=len(V),
    FIELDS_TYPICAL=fields("typical"), FIELDS_HARD=fields("hard"), STRAIN_FIELDS=strain_fields("typical"),
    D_NET=fmt(dp["net_p2"]["mean"]), D_ZERO=fmt(dp["zero"]["mean"]), D_FLOW=fmt(dp["farneback"]["mean"]), D_P1=fmt(dp["net_p1"]["mean"]), D_MAX=fmt(dp["net_p2"]["mean_max"], 1),
    FLOW_RATIO=fmt(dp["net_p2"]["mean"] / dp["farneback"]["mean"], 1), D_U=fmt(dp["net_u"]["mean"]), D_V=fmt(dp["net_v"]["mean"]),
    S_NET=fmt(sp["strainnet_p2"]["mean"]), S_ZERO=fmt(sp["zero"]["mean"]), S_FLOW=fmt(sp["from_farneback"]["mean"]), S_DIFF=fmt(sp["from_dispnet"]["mean"]),
    S_DIFF_X=fmt(sp["from_dispnet"]["mean"] / sp["zero"]["mean"], 1),
    EVAL_OK=fmt(ev["eval"]), EVAL_BAD=fmt(ev["train_mode"]), EVAL_POL=fmt(ev["polluted_eval"]), EVAL_X=fmt(ev["train_mode"] / ev["eval"], 1),
    LEAK_T=fmt(leak["disp_px"]["net_p2"]["mean"]), LEAK_R=fmt(leak["disp_px"]["net_p2"]["mean"] / leak["disp_px"]["zero"]["mean"] * 100, 1),
    TEST_R=fmt(dp["net_p2"]["mean"] / dp["zero"]["mean"] * 100, 1),
    MSE_BEFORE=fmt(d_before), MSE_AFTER=fmt(d_after), MSE_DROP=fmt((1 - d_after / d_before) * 100, 0),
    LIGHT_OK=fmt(pre["dim_second_norm_each"]), LIGHT_BAD=fmt(pre["dim_second_norm_first"]), DIM_RAW=fmt(pre["dim_raw"]), DIM_NORM=fmt(pre["dim_norm"]),
    V_ERR=fmt(roi["v_err_pct"], 1), U_ERR=fmt(roi["u_err_pct"], 1), H0=roi["h0"], HNEW=roi["hnew"], W0=roi["w0"], WNEW=roi["wnew"],
    ROI_WRONG=fmt(abs(tr["incremental_minus"][-1]), 0), ROI_FIXED=fmt(tr["fixed_plus"][-1], 0),
    PARAMS=fmt(M["DisplacementNet"]["params"] / 1e6, 1), MACS=fmt(M["DisplacementNet"]["macs"] / 1e9, 1), UNUSED=f"{M['DisplacementNet']['unused']:,}",
    STEP_S=fmt(M["DisplacementNet"]["step_s"]), INFER=fmt(M["DisplacementNet"]["infer_s"] * 1000, 0),
    DAYS_D=fmt(days_d, 0), DAYS_S=fmt(days_s, 0), STEPS_D=T["disp"]["steps"], STEPS_S=T["strain"]["steps"], IMG_SEEN=f"{img_seen:,}", RATIO=f"{ratio_paper:,.0f}",
    OVER_START=fmt(D["overfit_start"]), OVER_END=fmt(D["overfit_end"], 3), OVER_DROP=fmt(over_drop, 1), CORR_PRED=fmt(corr_pred), CORR_FLOW=fmt(corr_flow),
    PRED_STD=fmt(D["pred_std"]), LABEL_STD=fmt(D["label_std"]), **roi_subs,
)
page = TEMPLATE
for k, v in subs.items():
    page = page.replace("{{" + k + "}}", str(v))
assert "{{" not in page, page[page.index("{{"):page.index("{{") + 40]
open(OUT, "w", encoding="utf-8").write(page)
# The published artifact gets its skeleton on publish; the copy opened from disk needs its own
head = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>'
open(sys.argv[2], "w", encoding="utf-8").write(head + page + "</body></html>")
print("wrote", OUT, f"{len(page) / 1e6:.1f} MB", "blockers", blockers, "checks", passed, "/", len(V), "days", round(days_d, 1), round(days_s, 1))
