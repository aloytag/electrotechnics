# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import RadioButtons

def plot_phasors_from_solution(solution,
                               currents=None,
                               voltages=None,
                               voltage_sources=None,
                               current_sources=None,
                               current_labels=None,
                               voltage_labels=None,
                               reference=None,
                               reference_mode='auto',   # 'auto' | 'current' | 'voltage'
                               # --- visual options ---
                               separate_scales=True,    # independent I/V scaling (visual only)
                               hide_axes=True,         # hide ticks, spines and labels
                               show_grid=False,        # no grid
                               arrow_width=2.2,        # arrow line width
                               arrow_head=14,          # arrow head size
                               tooltip_fontsize=14,    # tooltip text size
                               hover_tol_px=10,        # hover detection tolerance in pixels
                               title='Phasor Diagram',
                               ax=None):
    """
    Draw current (red) and voltage (blue) phasors with hover tooltips,
    independent visual scales, and group-toggle metadata.

    Parameters
    ----------
    solution : dict
        Solution dictionary from solve_circuit() containing node_voltages and branch_currents.
    currents : list, optional
        Current phasors to plot, specified as:
        - int k: k-th branch current
        - ('I', k): k-th branch current (alternative syntax)
        - ('vs', j): j-th voltage source current
        - ('is', j): j-th current source value
    voltages : list, optional
        Voltage phasors to plot, specified as:
        - (n_plus, n_minus): node voltage difference
        - ('vs', j): j-th voltage source value
    voltage_sources : list of tuples, optional
        List of voltage sources (required for 'vs' specifications).
    current_sources : list of tuples, optional
        List of current sources (required for 'is' specifications).
    current_labels, voltage_labels : list of str, optional
        Custom labels for currents and voltages.
    reference : str | tuple | complex | None
        Angular reference for phasor rotation:
        - str: existing phasor label
        - tuple: same specification as currents/voltages
        - complex: explicit reference value
        - None: uses first voltage or current
    reference_mode : {'auto', 'current', 'voltage'}
        Disambiguation for ('vs', j) reference specification.
    separate_scales : bool, default=True
        If True, scales largest |I| to match largest |V| visually.
        Only affects drawing; tooltips show actual unscaled values.
    hide_axes : bool, default=True
        If True, hides ticks, spines and labels.
    show_grid : bool, default=False
        If True, shows grid lines.
    arrow_width : float, default=2.2
        Width of phasor arrows.
    arrow_head : float, default=14
        Size of arrow heads.
    tooltip_fontsize : float, default=14
        Font size for hover tooltips.
    hover_tol_px : float, default=10
        Hover detection radius in pixels.
    title : str, default='Phasor Diagram'
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    dict
        Plot configuration with keys:
        - 'fig': matplotlib Figure
        - 'ax': matplotlib Axes
        - 'ref': reference phasor value
        - 'scales': scaling factors used
        - 'artists': list of plotted arrows
        - 'tooltip': tooltip annotation
        - 'mpl_cid': callback connection id
        - 'group_indices': indices for current/voltage groups
    """
    Vnodes = solution.get('node_voltages', None)
    Ibranches = solution.get('branch_currents', None)
    I_vs = solution.get('voltage_source_currents', np.zeros((0,), dtype=complex))

    currents = currents or []
    voltages = voltages or []
    current_labels = current_labels or []
    voltage_labels = voltage_labels or []

    # ---- Build lists (label, complex phasor, type) ----
    cur_list = []
    for idx, item in enumerate(currents):
        if isinstance(item, int):
            if Ibranches is None:
                raise ValueError("No 'branch_currents'. Call solve_circuit_mna(..., return_branch_currents=True).")
            z = Ibranches[item]
            label = current_labels[idx] if idx < len(current_labels) else f"I_b{item}"
        elif isinstance(item, tuple) and len(item) == 2:
            tag, val = item
            if tag == 'I':
                if Ibranches is None:
                    raise ValueError("No 'branch_currents'. Enable return_branch_currents=True.")
                z = Ibranches[val]
                label = current_labels[idx] if idx < len(current_labels) else f"I_b{val}"
            elif tag == 'vs':
                z = I_vs[val]
                label = current_labels[idx] if idx < len(current_labels) else f"I_vs{val}"
            elif tag == 'is':
                if current_sources is None:
                    raise ValueError("For ('is', j) you need current_sources.")
                z = current_sources[val][2]
                label = current_labels[idx] if idx < len(current_labels) else f"I_is{val}"
            else:
                raise ValueError(f"Unrecognized current specification: {item}")
        else:
            raise ValueError(f"Unrecognized current specification: {item}")
        cur_list.append((label, z, 'I'))

    volt_list = []
    for idx, item in enumerate(voltages):
        if isinstance(item, tuple) and len(item) == 2 and all(isinstance(v, int) for v in item):
            n_plus, n_minus = item
            if Vnodes is None:
                raise ValueError("No hay 'node_voltages' en solution.")
            z = Vnodes[n_plus] - Vnodes[n_minus]
            label = voltage_labels[idx] if idx < len(voltage_labels) else f"V_{n_plus}-{n_minus}"
        elif isinstance(item, tuple) and len(item) == 2 and item[0] == 'vs':
            j = item[1]
            if voltage_sources is None:
                raise ValueError("For ('vs', j) in voltages, provide voltage_sources.")
            z = voltage_sources[j][2]  # E
            label = voltage_labels[idx] if idx < len(voltage_labels) else f"E_vs{j}"
        else:
            raise ValueError(f"Unrecognized voltage specification: {item}")
        volt_list.append((label, z, 'V'))

    # Sets for 'auto' reference
    spec_currents_set = {('I', it) if isinstance(it, int) else tuple(['I'] + list(it)) for it in currents}
    spec_voltages_set = {('V', it) for it in voltages}

    def _get_phasor_from_spec(spec):
        # existing label
        if isinstance(spec, str):
            for L, z, t in (cur_list + volt_list):
                if L == spec:
                    return z
            raise ValueError(f"Label '{spec}' not found.")
        # explicit complex number
        if isinstance(spec, complex) or isinstance(spec, np.complexfloating):
            return complex(spec)
        # integer -> branch current
        if isinstance(spec, int):
            if Ibranches is None:
                raise ValueError("No 'branch_currents' for branch reference.")
            return Ibranches[spec]
        # tupla
        if isinstance(spec, tuple) and len(spec) == 2:
            tag, val = spec
            if tag == 'I':
                if Ibranches is None:
                    raise ValueError("No hay 'branch_currents' para referencia por rama.")
                return Ibranches[val]
            if tag == 'is':
                if current_sources is None:
                    raise ValueError("For ('is', j) you need current_sources.")
                return current_sources[val][2]
            if tag == 'vs':
                if reference_mode == 'current':
                    return I_vs[val]
                if reference_mode == 'voltage':
                    if voltage_sources is None:
                        raise ValueError("'voltage' was requested with ('vs', j) but voltage_sources was not provided.")
                    return voltage_sources[val][2]
                # auto
                if ('I', ('vs', val)) in spec_currents_set:
                    return I_vs[val]
                if ('V', ('vs', val)) in spec_voltages_set:
                    if voltage_sources is None:
                        raise ValueError("('vs', j) is in voltages but voltage_sources was not provided.")
                    return voltage_sources[val][2]
                # by convention, prioritize voltage
                return voltage_sources[val][2] if voltage_sources is not None else I_vs[val]
            # (n+, n-) -> nodal voltage
            if all(isinstance(v, int) for v in spec):
                n_plus, n_minus = spec
                if Vnodes is None:
                    raise ValueError("No 'node_voltages' for voltage reference.")
                return Vnodes[n_plus] - Vnodes[n_minus]
        raise ValueError(f"Invalid reference specification: {spec}")

    # --- angular reference ---
    if reference is None:
        ref = volt_list[0][1] if len(volt_list) else (cur_list[0][1] if len(cur_list) else None)
        if ref is None:
            raise ValueError("No phasors to plot and no reference defined.")
    else:
        ref = _get_phasor_from_spec(reference)
    ang_ref = 0.0 if ref == 0 else np.angle(ref)
    rot = lambda z: z * np.exp(-1j * ang_ref)

    rot_currents = [(L, rot(z), t) for (L, z, t) in cur_list]
    rot_voltages = [(L, rot(z), t) for (L, z, t) in volt_list]

    # --- independant scales ---
    def _max_mag(items): return max([abs(z) for (_, z, _) in items], default=0.0)
    s_V = 1.0
    s_I = 1.0
    if separate_scales:
        maxV = _max_mag(rot_voltages)
        maxI = _max_mag(rot_currents)
        if maxV > 0 and maxI > 0:
            s_I = maxV / maxI

    # Axes setup
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        created_ax = True
    else:
        fig = ax.figure

    ax.set_title(title, fontsize=12)
    if show_grid is True:
        ax.grid(show_grid, ls=':', alpha=0.6)
    else:
        ax.grid(False)
    if hide_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xlabel(''); ax.set_ylabel('')

    # ---- drwing arrows with FancyArrowPatch ----
    artists = []   # dict: {'patch':arr, 'label':str, 'z_text':complex, 'color':str, 'end_xy':np.array, 'kind':'I'|'V'}
    def _add_arrow(z_draw, color, label, z_text, kind):
        arr = FancyArrowPatch((0, 0), (z_draw.real, z_draw.imag),
                              arrowstyle='->', mutation_scale=arrow_head,
                              color=color, linewidth=arrow_width)
        ax.add_patch(arr)
        end_xy = np.array([z_draw.real, z_draw.imag])
        artists.append({'patch': arr, 'label': label, 'z_text': z_text,
                        'color': color, 'end_xy': end_xy, 'kind': kind})

    for (L, z, _ ) in rot_currents:
        _add_arrow(z * s_I, 'red',  L, z, 'I')   # z_text = z (NO escalado)
    for (L, z, _ ) in rot_voltages:
        _add_arrow(z * s_V, 'blue', L, z, 'V')   # z_text = z (NO escalado)

    # ---- autoscale ----
    all_xy = [it['end_xy'] for it in artists] + [np.array([0.0, 0.0])]
    if len(all_xy):
        r = np.max(np.linalg.norm(all_xy, axis=1))
        lim = 1.15 * max(r, 1e-12)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')

    # Minimal legend
    ax.plot([], [], color='red', lw=3, label='Currents')
    ax.plot([], [], color='blue', lw=3, label='Voltages')
    ax.legend(loc='upper right')

    # ---- Interactive tooltip (single reusable Annotation) ----
    tooltip = ax.annotate("", xy=(0,0), xytext=(15, 15), textcoords="offset points",
                          bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.4", alpha=0.95),
                          arrowprops=dict(arrowstyle="->", color="0.4"),
                          fontsize=tooltip_fontsize)
    tooltip.set_visible(False)

    # utilities to convert pixel tolerance to data distance
    def _pix_to_data_radius(ax, px=10):
        inv = ax.transData.inverted()
        p0 = inv.transform((0,0))
        p1 = inv.transform((px,px))
        return np.mean(np.abs(p1 - p0))

    pick_rad = _pix_to_data_radius(ax, hover_tol_px)

    def _fmt_phasor_text(label, z):
        mag = abs(z)
        ang = np.angle(z, deg=True)
        return f"{label}\n|{mag:.4g}| ∠ {ang:.2f}°"

    def on_move(event):
        if not event.inaxes == ax or event.xdata is None or event.ydata is None:
            tooltip.set_visible(False)
            fig.canvas.draw_idle()
            return
        p = np.array([event.xdata, event.ydata])

        d2_min = None
        hit = None
        # IGNORE invisible artists
        for it in artists:
            if not it['patch'].get_visible():
                continue
            d2 = np.sum((p - it['end_xy'])**2)
            if (d2_min is None) or (d2 < d2_min):
                d2_min = d2; hit = it

        if hit is None or d2_min is None:
            tooltip.set_visible(False)
            fig.canvas.draw_idle()
            return

        if np.sqrt(d2_min) <= pick_rad:
            tooltip.xy = hit['end_xy']
            tooltip.set_text(_fmt_phasor_text(hit['label'], hit['z_text']))
            tooltip.get_bbox_patch().set_edgecolor(hit['color'])
            tooltip.set_visible(True)
        else:
            tooltip.set_visible(False)

        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("motion_notify_event", on_move)

    # group indices for toggles
    idx_I = [i for i, it in enumerate(artists) if it['kind'] == 'I']
    idx_V = [i for i, it in enumerate(artists) if it['kind'] == 'V']

    out = {
        'fig': fig,
        'ax': ax,
        'ref': ref,
        'scales': {'s_I': s_I, 's_V': s_V, 'separate_scales': separate_scales},
        'artists': artists,     # lista de dicts
        'tooltip': tooltip,
        'mpl_cid': cid,         # id de conexión del callback
        'group_indices': {'currents': idx_I, 'voltages': idx_V},
    }
    if created_ax:
        plt.tight_layout()
    return out

def attach_matplotlib_toggles(ph_out,
                              show_currents=True,
                              show_voltages=True,
                              box_pos=(0.02, 0.70, 0.18, 0.15),  # [left, bottom, width, height] in fig coords
                              labels=('Voltages', 'Currents')):
    """
    Add a CheckButtons panel to show/hide voltage and current phasors.

    Parameters
    ----------
    ph_out : dict
        Output dictionary from plot_phasors_from_solution().
    show_currents : bool, default=True
        Initial visibility of current phasors.
    show_voltages : bool, default=True
        Initial visibility of voltage phasors.
    box_pos : tuple(float, float, float, float), default=(0.02, 0.70, 0.18, 0.15)
        Panel position in figure coordinates (0 to 1): [left, bottom, width, height].
    labels : tuple(str, str), default=('Voltages', 'Currents')
        Labels for voltage and current toggles.

    Returns
    -------
    dict
        Toggle configuration with keys:
        - 'checkbuttons': matplotlib CheckButtons widget
        - 'ax': matplotlib Axes for the panel
    """
    fig = ph_out['fig']
    ax = ph_out['ax']
    artists = ph_out['artists']
    g = ph_out['group_indices']

    # estado inicial
    for i in g['voltages']:
        artists[i]['patch'].set_visible(show_voltages)
    for i in g['currents']:
        artists[i]['patch'].set_visible(show_currents)
    fig.canvas.draw_idle()

    # eje para checkbuttons
    check_ax = fig.add_axes(box_pos)
    checks = CheckButtons(check_ax, labels=labels,
                          actives=(show_voltages, show_currents))

    def on_clicked(label):
        if label == labels[0]:  # tensiones
            vis = not artists[g['voltages'][0]]['patch'].get_visible() if g['voltages'] else False
            for i in g['voltages']:
                artists[i]['patch'].set_visible(vis)
        elif label == labels[1]:  # corrientes
            vis = not artists[g['currents'][0]]['patch'].get_visible() if g['currents'] else False
            for i in g['currents']:
                artists[i]['patch'].set_visible(vis)
        fig.canvas.draw_idle()

    checks.on_clicked(on_clicked)

    return {'checkbuttons': checks, 'ax': check_ax}

def attach_matplotlib_order_widget(ph_out,
                                   initial='voltages-top',   # 'voltages-top' | 'currents-top'
                                   box_pos=(0.02, 0.52, 0.28, 0.13),
                                   labels=('Voltages on top', 'Currents on top'),
                                   z_bottom=2.0,
                                   z_top=3.0,
                                   legend_sync=False):
    """
    Add RadioButtons to choose whether VOLTAGES (blue) or CURRENTS (red)
    are drawn 'on top' (higher zorder) in the figure.

    Parameters
    ----------
    ph_out : dict
        Output from plot_phasors_from_solution(...). Must contain:
          - 'fig', 'ax'
          - 'artists': list of dicts with keys: 'patch', 'kind' ('I'|'V'), ...
          - 'group_indices': {'currents': [...], 'voltages': [...]}
          - 'tooltip' (Annotation), optional (will be moved to high zorder)
    initial : {'voltages-top', 'currents-top'}, default='voltages-top'
        Initial drawing order state.
    box_pos : tuple(float, float, float, float), default=(0.02, 0.52, 0.28, 0.13)
        Panel position in FIGURE coordinates (0 to 1): (left, bottom, width, height).
    labels : tuple(str, str), default=('Voltages on top', 'Currents on top')
        Labels for the two RadioButton options.
    z_bottom, z_top : float, default=2.0, 3.0
        zorder values used for 'bottom' and 'top' groups.
    legend_sync : bool, default=False
        If True, attempts to reorder legend so first item corresponds to 'top' group.

    Returns
    -------
    dict
        Widget configuration with keys:
          - 'radio': RadioButtons widget
          - 'ax': panel axes
          - 'get_mode': function returning current mode ('voltages-top'|'currents-top')
          - 'set_mode': function to programmatically change mode
    """
    fig = ph_out['fig']
    ax_main = ph_out['ax']
    artists = ph_out['artists']
    groups = ph_out['group_indices']
    tooltip = ph_out.get('tooltip', None)

    # Always keep tooltip on top of everything
    if tooltip is not None:
        tooltip.set_zorder(1000.0)

    # Helpers internos para setear zorder por grupo
    def _set_group_zorder(group_name, z):
        for i in groups.get(group_name, []):
            artists[i]['patch'].set_zorder(z)

    def _apply_mode(mode):
        if mode == 'voltages-top':
            _set_group_zorder('voltages', z_top)
            _set_group_zorder('currents', z_bottom)
        elif mode == 'currents-top':
            _set_group_zorder('voltages', z_bottom)
            _set_group_zorder('currents', z_top)
        else:
            raise ValueError("Modo no reconocido (usar 'voltages-top' o 'currents-top').")

        # Opcional: reordenar leyenda
        if legend_sync:
            # reconstruye leyenda: primero el grupo 'top'
            handles, labels_legend = ax_main.get_legend_handles_labels()
            # In our figure, legend items are 2 dummy lines (red/blue).
            # To make legend reflect the order, reorder them:
            if len(handles) >= 2:
                if mode == 'voltages-top':
                    # blue first, red second
                    order = [1, 0] if labels_legend[0].lower().startswith('curr') else [0, 1]
                else:
                    # red first, blue second
                    order = [0, 1] if labels_legend[0].lower().startswith('curr') else [1, 0]
                try:
                    ax_main.legend([handles[i] for i in order],
                                   [labels_legend[i] for i in order],
                                   loc='upper right')
                except Exception:
                    pass

        fig.canvas.draw_idle()

    # RadioButtons
    rax = fig.add_axes(box_pos)
    radio = RadioButtons(rax, (labels[0], labels[1]))

    # Initial widget state from 'initial' argument
    if initial == 'voltages-top':
        radio.set_active(0)  # first label
    elif initial == 'currents-top':
        radio.set_active(1)
    else:
        raise ValueError("initial debe ser 'voltages-top' o 'currents-top'.")

    _apply_mode(initial)

    # Callback
    def _on_clicked(label):
        if label == labels[0]:
            _apply_mode('voltages-top')
        elif label == labels[1]:
            _apply_mode('currents-top')

    radio.on_clicked(_on_clicked)

    # API
    def get_mode():
        # Returns the z-order current mode.
        active = radio.value_selected
        return 'voltages-top' if active == labels[0] else 'currents-top'

    def set_mode(mode):
        # state programmatic change
        if mode == 'voltages-top':
            radio.set_active(0)
            _apply_mode('voltages-top')
        elif mode == 'currents-top':
            radio.set_active(1)
            _apply_mode('currents-top')
        else:
            raise ValueError("Modo no reconocido.")

    return {'radio': radio, 'ax': rax, 'get_mode': get_mode, 'set_mode': set_mode}

def phasor_plot(solution,
                currents=None,
                voltages=None,
                voltage_sources=None,
                current_sources=None,
                current_labels=None,
                voltage_labels=None,
                reference=None,
                reference_mode='auto',   # 'auto' | 'current' | 'voltage'
                # --- visual options ---
                separate_scales=True,    # independent I/V scaling (visual only)
                hide_axes=True,          # hide ticks, spines and labels
                show_grid=False,         # no grid
                arrow_width=2.2,         # arrow line width
                arrow_head=14,           # arrow head size
                tooltip_fontsize=14,     # tooltip text size
                hover_tol_px=10,         # hover detection tolerance in pixels
                title='Phasor Diagram',
                ax=None):
    
    ph = plot_phasors_from_solution(
        solution=solution,
        currents=currents,
        voltages=voltages,
        voltage_sources=voltage_sources,
        current_sources=current_sources,
        current_labels=current_labels,
        voltage_labels=voltage_labels,
        reference=reference,
        reference_mode=reference_mode,
        separate_scales=separate_scales,
        hide_axes=hide_axes,
        show_grid=show_grid,
        arrow_width=arrow_width,
        arrow_head=arrow_head,
        tooltip_fontsize=tooltip_fontsize,
        hover_tol_px=hover_tol_px,
        title=title,
        ax=ax
    )

    tog = attach_matplotlib_toggles(ph)
    ordw = attach_matplotlib_order_widget(ph_out=ph,
                                   initial='voltages-top',
                                   box_pos=(0.02, 0.52, 0.30, 0.13),
                                   labels=('Voltages on top', 'Currents on top'),
                                   z_bottom=2.0, z_top=3.0, legend_sync=False)
    
    if not plt.isinteractive():
        plt.show()
