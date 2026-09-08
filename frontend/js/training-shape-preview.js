// Weight-decomposition preview for the Anima training network; composed into the
// training UI by app.js. Draws the current module/algo as schematic matrices and
// reports the trainable parameter count for one example linear layer.
//
// Shapes mirror the vendored implementations exactly:
//   LoRA  — sd-scripts networks/lora_anima.py, LyCORIS modules/locon.py
//   LoHa  — sd-scripts networks/loha.py, LyCORIS modules/loha.py
//   LoKr  — sd-scripts networks/lokr.py, LyCORIS modules/lokr.py
//   factor() — vendor/lycoris/functional/general.py (identical to sd-scripts)
window.trainingShapePreviewMixin = {
  shapePreviewOpen: false,
  shapePreviewPreviousFocus: null,

  // 示例层固定为 2048×2048 线性层：只用于展示形状与参数量关系，
  // 不声称对应模型中的具体层。
  SHAPE_PREVIEW_LAYER: { in: 2048, out: 2048 },

  shapePreviewSupported() {
    return String((this.form && this.form.model_train_type) || '') === 'anima-lora';
  },

  // 入口按钮由 training-core.js 的 _getEnvHint('network_module') 追加到字段下方。
  _shapePreviewEntry() {
    if (!this.shapePreviewSupported()) return '';
    return `<div class="shape-preview-entry">
      <button type="button" class="btn btn-ghost btn-sm" @click="openShapePreview()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/></svg>
        <span x-text="t('shapePreview.open')">Structure preview</span>
      </button>
    </div>`;
  },

  openShapePreview() {
    if (!this.shapePreviewSupported()) return;
    this._openManagedModal('shapePreviewOpen', 'shapePreviewPreviousFocus', '.shape-preview-close');
  },

  closeShapePreview() {
    this._closeManagedModal('shapePreviewOpen', 'shapePreviewPreviousFocus');
  },

  _shapePreviewNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  },

  _shapePreviewInt(value, fallback) {
    const parsed = Math.round(this._shapePreviewNumber(value, fallback));
    return parsed > 0 ? parsed : fallback;
  },

  // 与 vendor/lycoris/functional/general.py factorization() 完全一致：
  // factor > 0 且整除时直接拆；否则从 1 起找使两块之和最小的因子。
  _shapePreviewFactorization(dimension, factor) {
    const dim = Math.max(1, Math.floor(Number(dimension) || 0));
    const limitFactor = Number(factor);
    const swap = (a, b) => (a > b ? [b, a] : [a, b]);
    if (limitFactor > 0 && dim % limitFactor === 0) {
      return swap(limitFactor, dim / limitFactor);
    }
    let limit = limitFactor < 0 ? dim : limitFactor;
    let m = 1;
    let n = dim;
    let length = m + n;
    while (m < n) {
      let newM = m + 1;
      while (dim % newM !== 0) newM += 1;
      const newN = dim / newM;
      if (newM + newN > length || newM > limit) break;
      m = newM;
      n = newN;
      length = m + n;
    }
    return swap(m, n);
  },

  _shapePreviewCount(value) {
    return Number(value).toLocaleString();
  },

  _shapePreviewScale(value) {
    if (!Number.isFinite(value)) return '—';
    return String(Math.round(value * 1000) / 1000);
  },

  _shapePreviewPct(trainable, full) {
    if (!(full > 0)) return '—';
    const pct = (trainable / full) * 100;
    return `${pct < 0.01 ? '<0.01' : pct.toFixed(pct < 1 ? 2 : 1)}%`;
  },

  // ── 数据：当前配置 → 形状、参数、说明 ──────────────────────
  _buildShapePreview() {
    const f = this.form || {};
    const module = String(f.network_module || '');
    if (!module) return null;
    const isLycoris = module === 'lycoris.kohya';
    const algo = isLycoris
      ? String(f.lycoris_algo || 'lora').toLowerCase()
      : (module === 'networks.loha' ? 'loha' : (module === 'networks.lokr' ? 'lokr' : 'lora'));
    const rank = this._shapePreviewInt(f.network_dim, 32);
    const alpha = this._shapePreviewInt(f.network_alpha, rank);
    const rsLora = isLycoris && f.rs_lora === true;
    const inDim = this.SHAPE_PREVIEW_LAYER.in;
    const outDim = this.SHAPE_PREVIEW_LAYER.out;
    const fullParams = inDim * outDim;
    const notes = [];
    const t = (key, fallback) => this.t(key, fallback);
    const fill = (key, values) => Object.keys(values)
      .reduce((text, name) => text.replace(`{${name}}`, String(values[name])), t(key));

    const algoLabel = isLycoris
      ? this._fieldOptionLabel('lycoris_algo', algo, algo)
      : (algo === 'loha' ? 'LoHa' : (algo === 'lokr' ? 'LoKr' : 'LoRA'));
    const meta = [
      { label: t('shapePreview.module'), value: module },
      { label: t('shapePreview.algo'), value: algoLabel },
      { label: t('shapePreview.rank'), value: String(rank) },
      { label: t('shapePreview.alpha'), value: String(alpha) },
    ];
    const legend = [t('shapePreview.legendMatMul')];

    let trainable = 0;
    let scaleText = fill(rsLora ? 'shapePreview.scaleFormulaRs' : 'shapePreview.scaleFormula', {
      alpha,
      rank,
      sqrtRank: this._shapePreviewScale(Math.sqrt(rank)),
      value: this._shapePreviewScale(rsLora ? alpha / Math.sqrt(rank) : alpha / rank),
    });
    let caption = [t('shapePreview.loraCaption')];
    let diagram = '';

    if (algo === 'lora' || algo === 'loha') {
      if (algo === 'loha') {
        trainable = 2 * rank * (inDim + outDim);
        caption = [t('shapePreview.lohaCaption')];
        legend.push(t('shapePreview.legendHadamard'));
        diagram = this._shapePreviewLohaSvg({ rank, inDim, outDim });
      } else {
        trainable = rank * (inDim + outDim);
        diagram = this._shapePreviewLoraSvg({ rank, inDim, outDim });
      }
      if (isLycoris && f.conv_dim !== '' && f.conv_dim !== null && f.conv_dim !== undefined) {
        notes.push(fill('shapePreview.convNote', { dim: f.conv_dim }));
      }
      if (isLycoris && f.use_scalar === true) notes.push(t('shapePreview.useScalarNote'));
      if (rsLora) notes.push(t('shapePreview.rsLoraNote'));
    } else {
      const factor = Math.round(this._shapePreviewNumber(f.lokr_factor, -1));
      const fullMatrix = isLycoris && f.full_matrix === true;
      const decomposeBoth = isLycoris && f.decompose_both === true;
      const unbalanced = isLycoris && f.unbalanced_factorization === true;
      const [inM, inN] = this._shapePreviewFactorization(inDim, factor);
      let [outL, outK] = this._shapePreviewFactorization(outDim, factor);
      if (unbalanced) { const swap = outL; outL = outK; outK = swap; }
      const w1LowRank = !fullMatrix && decomposeBoth && rank < Math.max(outL, inM) / 2;
      // 与训练端一致：rank 未低于第二块尺寸的一半时，w2 直接存完整矩阵。
      const w2Full = fullMatrix || !(rank < Math.max(outK, inN) / 2);
      const w1Params = w1LowRank ? outL * rank + rank * inM : outL * inM;
      const w2Params = w2Full ? outK * inN : outK * rank + rank * inN;
      trainable = w1Params + w2Params;
      // 训练端实际是 ΔW = kron(w1, w2)，w2 为低秩时先乘出 w2 再参与 Kronecker 积，
      // 所以低秩对必须加括号，否则按优先级会被读成 (W1 ⊗ W2a) × W2b。
      caption = [
        fill('shapePreview.kronCaption', { outL, outK, inM, inN }),
        fill('shapePreview.kronFormula', {
          w1: w1LowRank ? '(W1a × W1b)' : 'W1',
          w2: w2Full ? 'W2' : '(W2a × W2b)',
        }),
      ];
      legend.push(t('shapePreview.legendKron'));
      diagram = this._shapePreviewLokrSvg({
        rank, inDim, outDim, outL, outK, inM, inN, w1LowRank, w2Full,
      });
      meta.push({
        label: t('shapePreview.factor'),
        value: factor < 0 ? `${factor} · ${t('shapePreview.factorAuto')}` : String(factor),
      });
      meta.push({
        label: t('shapePreview.w1'),
        value: w1LowRank
          ? `${t('shapePreview.lowRank')} W1a ${outL} × ${rank} · W1b ${rank} × ${inM}`
          : `${t('shapePreview.fullMatrix')} ${outL} × ${inM}`,
      });
      meta.push({
        label: t('shapePreview.w2'),
        value: w2Full
          ? `${t('shapePreview.fullMatrix')} ${outK} × ${inN}`
          : `${t('shapePreview.lowRank')} W2a ${outK} × ${rank} · W2b ${rank} × ${inN}`,
      });
      notes.push(fill(factor < 0 ? 'shapePreview.factorAutoNote' : 'shapePreview.factorFixedNote', {
        factor, outL, outK, inM, inN,
      }));
      if (w2Full && !fullMatrix) {
        notes.push(fill('shapePreview.w2AutoFullNote', { rank, outK, inN }));
      } else if (fullMatrix) {
        notes.push(t('shapePreview.w2FullMatrixNote'));
      }
      if (decomposeBoth) {
        notes.push(fill(
          w1LowRank ? 'shapePreview.w1LowRankNote' : 'shapePreview.w1StaysFullNote',
          { rank, threshold: Math.max(outL, inM) / 2 }
        ));
      }
      if (unbalanced) notes.push(t('shapePreview.unbalancedNote'));
      if (rsLora) notes.push(t('shapePreview.rsLoraNote'));
      if (isLycoris && f.use_scalar === true) notes.push(t('shapePreview.useScalarNote'));
      if (w2Full && !w1LowRank) {
        // LyCORIS 在 w1、w2 都是完整矩阵时把 alpha 固定为 rank；原生 LoKr 的 w1 恒为完整矩阵。
        scaleText = t('shapePreview.scaleForced');
      }
    }

    if (isLycoris && f.dora_wd === true) {
      const axis = f.wd_on_output === false ? t('shapePreview.axisIn') : t('shapePreview.axisOut');
      notes.push(fill('shapePreview.doraNote', {
        count: this._shapePreviewCount(f.wd_on_output === false ? inDim : outDim),
        axis,
      }));
    }
    if (isLycoris && f.train_llm_adapter === true) notes.push(t('shapePreview.llmAdapterNote'));
    if (!isLycoris && f.train_adaln === true) notes.push(t('shapePreview.adalnNote'));

    meta.push({ label: t('shapePreview.scale'), value: scaleText });
    meta.push({
      label: t('shapePreview.exampleLayer'),
      value: `${this._shapePreviewCount(inDim)} × ${this._shapePreviewCount(outDim)}`,
    });
    meta.push({
      label: t('shapePreview.trainableParams'),
      value: this._shapePreviewCount(trainable)
        + fill('shapePreview.ratio', { value: this._shapePreviewPct(trainable, fullParams) }),
    });
    meta.push({
      label: t('shapePreview.fullParams'),
      value: this._shapePreviewCount(fullParams),
    });

    return { meta, notes, caption, diagram, legend };
  },

  // ── SVG 示意：尺寸按 log 压缩，数字才是准确值 ───────────────
  _shapePreviewSize(dim) {
    const value = Math.max(2, this._shapePreviewNumber(dim, 2));
    const norm = Math.min(1, Math.log2(value) / 13);
    return 26 + norm * norm * 124;
  },

  _shapePreviewBlockSvg(x, y, w, h, cls) {
    return `<rect class="sp-block ${cls}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="3"/>`;
  },

  _shapePreviewTextSvg(x, y, text, cls) {
    return `<text class="sp-${cls}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="middle">${this.esc(String(text))}</text>`;
  },

  // 低秩对的成组框：虚线圆角框把「这两块先相乘，再作为整体参与 ⊗」圈在一起。
  // 框要包住块上方的名称与下方的维度标签，所以上下各留 20/22 单位。
  _shapePreviewGroupSvg(left, right, top, bottom) {
    const boxLeft = left - 14;
    const boxRight = right + 14;
    const boxTop = top - 20;
    const boxBottom = bottom + 22;
    return `<rect class="sp-group" x="${boxLeft.toFixed(1)}" y="${boxTop.toFixed(1)}"`
      + ` width="${(boxRight - boxLeft).toFixed(1)}" height="${(boxBottom - boxTop).toFixed(1)}" rx="8"/>`;
  },

  _shapePreviewMatrixSvg(x, y, w, h, cls, name, dimText) {
    return this._shapePreviewBlockSvg(x, y, w, h, cls)
      + this._shapePreviewTextSvg(x + w / 2, y - 7, name, 'name')
      + this._shapePreviewTextSvg(x + w / 2, y + h + 15, dimText, 'dim');
  },

  _shapePreviewSvgWrap(used, top, bottom, parts) {
    const pad = 26;
    const width = Math.max(140, used) + pad * 2;
    const height = Math.max(60, bottom - top) + pad * 2;
    return `<svg class="shape-diagram-svg" viewBox="0 0 ${width.toFixed(1)} ${height.toFixed(1)}" preserveAspectRatio="xMidYMid meet" role="img"`
      + ` aria-label="${this.escapeAttr(this.t('shapePreview.diagramAlt'))}">`
      + `<g transform="translate(${pad},${(pad - top).toFixed(1)})">${parts.join('')}</g></svg>`;
  },

  _shapePreviewLoraSvg({ rank, inDim, outDim }) {
    const cy = 0;
    const bw = this._shapePreviewSize(rank);
    const bh = this._shapePreviewSize(outDim);
    const aw = this._shapePreviewSize(inDim);
    const ah = this._shapePreviewSize(rank);
    const dw = this._shapePreviewSize(inDim);
    const dh = this._shapePreviewSize(outDim);
    const gap = 44;
    const parts = [];
    let x = 0;
    parts.push(this._shapePreviewMatrixSvg(x, cy - bh / 2, bw, bh, 'sp-block--b', 'B', `${outDim} × ${rank}`));
    x += bw + gap;
    parts.push(this._shapePreviewTextSvg(x, cy + 6, '×', 'op'));
    x += gap;
    parts.push(this._shapePreviewMatrixSvg(x, cy - ah / 2, aw, ah, 'sp-block--a', 'A', `${rank} × ${inDim}`));
    x += aw + gap;
    parts.push(this._shapePreviewTextSvg(x, cy + 6, '=', 'op'));
    x += gap;
    parts.push(this._shapePreviewMatrixSvg(x, cy - dh / 2, dw, dh, 'sp-block--out', 'ΔW', `${outDim} × ${inDim}`));
    x += dw;
    const half = Math.max(bh, dh) / 2;
    return this._shapePreviewSvgWrap(x, cy - half - 12, cy + half + 22, parts);
  },

  _shapePreviewLohaSvg({ rank, inDim, outDim }) {
    const rows = [0, 168];
    const bw = this._shapePreviewSize(rank);
    const bh = this._shapePreviewSize(outDim);
    const aw = this._shapePreviewSize(inDim);
    const ah = this._shapePreviewSize(rank);
    const dw = this._shapePreviewSize(inDim);
    const dh = this._shapePreviewSize(outDim);
    const gap = 40;
    const parts = [];
    let used = 0;
    rows.forEach((cy, index) => {
      let x = 0;
      parts.push(this._shapePreviewMatrixSvg(x, cy - bh / 2, bw, bh, 'sp-block--b', `B${index + 1}`, `${outDim} × ${rank}`));
      x += bw + gap;
      parts.push(this._shapePreviewTextSvg(x, cy + 6, '×', 'op'));
      x += gap;
      parts.push(this._shapePreviewMatrixSvg(x, cy - ah / 2, aw, ah, 'sp-block--a', `A${index + 1}`, `${rank} × ${inDim}`));
      used = Math.max(used, x + aw);
    });
    const midY = (rows[0] + rows[1]) / 2;
    let x = used + gap;
    parts.push(this._shapePreviewTextSvg(x, midY + 7, '⊙', 'op'));
    x += gap;
    parts.push(this._shapePreviewTextSvg(x, midY + 6, '=', 'op'));
    x += gap;
    parts.push(this._shapePreviewMatrixSvg(x, midY - dh / 2, dw, dh, 'sp-block--out', 'ΔW', `${outDim} × ${inDim}`));
    x += dw;
    return this._shapePreviewSvgWrap(x, rows[0] - bh / 2 - 12, rows[1] + bh / 2 + 22, parts);
  },

  _shapePreviewLokrSvg({ rank, inDim, outDim, outL, outK, inM, inN, w1LowRank, w2Full }) {
    const cy = 0;
    const gap = 40;
    const parts = [];
    const groups = [];
    let maxHalf = 0;
    // 成组框比块再宽 14，第一组左侧留出这段空间。
    let x = w1LowRank ? 14 : 0;

    if (w1LowRank) {
      const b1w = this._shapePreviewSize(rank);
      const b1h = this._shapePreviewSize(outL);
      const a1w = this._shapePreviewSize(inM);
      const a1h = this._shapePreviewSize(rank);
      maxHalf = Math.max(maxHalf, b1h / 2, a1h / 2);
      const pairTop = cy - Math.max(b1h, a1h) / 2;
      const pairBottom = cy + Math.max(b1h, a1h) / 2;
      const pairStart = x;
      parts.push(this._shapePreviewMatrixSvg(x, cy - b1h / 2, b1w, b1h, 'sp-block--b', 'W1a', `${outL} × ${rank}`));
      x += b1w + gap;
      parts.push(this._shapePreviewTextSvg(x, cy + 6, '×', 'op'));
      x += gap;
      parts.push(this._shapePreviewMatrixSvg(x, cy - a1h / 2, a1w, a1h, 'sp-block--a', 'W1b', `${rank} × ${inM}`));
      x += a1w;
      groups.push({ left: pairStart, right: x, top: pairTop, bottom: pairBottom });
    } else {
      const w1w = this._shapePreviewSize(inM);
      const w1h = this._shapePreviewSize(outL);
      maxHalf = Math.max(maxHalf, w1h / 2);
      parts.push(this._shapePreviewMatrixSvg(x, cy - w1h / 2, w1w, w1h, 'sp-block--b', 'W1', `${outL} × ${inM}`));
      x += w1w;
    }

    x += gap;
    parts.push(this._shapePreviewTextSvg(x, cy + 7, '⊗', 'op'));
    x += gap;

    if (w2Full) {
      const w2w = this._shapePreviewSize(inN);
      const w2h = this._shapePreviewSize(outK);
      maxHalf = Math.max(maxHalf, w2h / 2);
      parts.push(this._shapePreviewMatrixSvg(x, cy - w2h / 2, w2w, w2h, 'sp-block--a', 'W2', `${outK} × ${inN}`));
      x += w2w;
    } else {
      const b2w = this._shapePreviewSize(rank);
      const b2h = this._shapePreviewSize(outK);
      const a2w = this._shapePreviewSize(inN);
      const a2h = this._shapePreviewSize(rank);
      maxHalf = Math.max(maxHalf, b2h / 2, a2h / 2);
      const pairTop = cy - Math.max(b2h, a2h) / 2;
      const pairBottom = cy + Math.max(b2h, a2h) / 2;
      const pairStart = x;
      // 训练端是 w2 = w2a @ w2b：a 是 (out_k × rank)，b 是 (rank × in_n)。
      parts.push(this._shapePreviewMatrixSvg(x, cy - b2h / 2, b2w, b2h, 'sp-block--a', 'W2a', `${outK} × ${rank}`));
      x += b2w + gap;
      parts.push(this._shapePreviewTextSvg(x, cy + 6, '×', 'op'));
      x += gap;
      parts.push(this._shapePreviewMatrixSvg(x, cy - a2h / 2, a2w, a2h, 'sp-block--a', 'W2b', `${rank} × ${inN}`));
      x += a2w;
      groups.push({ left: pairStart, right: x, top: pairTop, bottom: pairBottom });
    }

    x += gap;
    parts.push(this._shapePreviewTextSvg(x, cy + 6, '=', 'op'));
    x += gap;
    const dw = this._shapePreviewSize(inDim);
    const dh = this._shapePreviewSize(outDim);
    maxHalf = Math.max(maxHalf, dh / 2);
    parts.push(this._shapePreviewMatrixSvg(x, cy - dh / 2, dw, dh, 'sp-block--out', 'ΔW', `${outDim} × ${inDim}`));
    x += dw;

    let minY = cy - maxHalf - 12;
    let maxY = cy + maxHalf + 22;
    let used = x;
    groups.forEach(group => {
      const boxLeft = group.left - 14;
      const boxRight = group.right + 14;
      const boxTop = group.top - 20;
      const boxBottom = group.bottom + 22;
      parts.push(this._shapePreviewGroupSvg(group.left, group.right, group.top, group.bottom));
      minY = Math.min(minY, boxTop - 8);
      maxY = Math.max(maxY, boxBottom + 8);
      used = Math.max(used, boxRight + 8);
    });
    return this._shapePreviewSvgWrap(used, minY, maxY, parts);
  },

  // ── 弹窗内容（x-html 注入，仅弹窗打开时求值）──────────────
  shapePreviewHtml() {
    const data = this._buildShapePreview();
    if (!data) {
      return `<div class="shape-preview-layout"><div class="shape-preview-sidebar">`
        + `<div class="shape-preview-notes"><div><span aria-hidden="true">•</span><span>${this.esc(this.t('shapePreview.unknown'))}</span></div></div>`
        + `</div><div class="shape-preview-diagram"></div></div>`;
    }
    const metaHtml = data.meta.map(row =>
      `<span><small>${this.esc(row.label)}</small><b>${this.esc(row.value)}</b></span>`).join('');
    const notesHtml = data.notes.length
      ? `<div class="shape-preview-notes">${data.notes.map(note =>
        `<div><span aria-hidden="true">•</span><span>${this.esc(note)}</span></div>`).join('')}</div>`
      : '';
    const legendHtml = data.legend.map(item => `<span>${this.esc(item)}</span>`).join('');
    const captionHtml = data.caption.map(line => `<div>${this.esc(line)}</div>`).join('');
    return `<div class="shape-preview-layout">
      <div class="shape-preview-sidebar">
        <div class="shape-preview-meta">${metaHtml}</div>
        ${notesHtml}
        <p class="shape-preview-footnote">${this.esc(this.t('shapePreview.footnote'))}</p>
      </div>
      <div class="shape-preview-diagram">
        <div class="shape-diagram-caption">${captionHtml}</div>
        ${data.diagram}
        <div class="shape-diagram-legend">${legendHtml}</div>
      </div>
    </div>`;
  },
};
