// Editor for MOC • LoRA Stack: rows of LoRA + strength, named presets in user/<user>/lora_stacks.
// The rows live in the node's hidden stack_data widget, so workflows keep their own copy;
// the linked preset name is a node property and never changes what runs.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STACK_NODE = "MocLoraStack";
const BY_NAME_NODE = "MocApplyLoraStackByName";
const DIR = "lora_stacks";
const SCHEMA_VERSION = 1;
const LINK_PROPERTY = "moc_stack_name";
const NAME_PART = /^[A-Za-z0-9_\-][A-Za-z0-9 _.\-]{0,99}$/;
const STRENGTH_LIMIT = 100;

const editors = new Set();
let loraList = null;

// ---------------------------------------------------------------- data

function validName(name) {
  const parts = name.split("/");
  return parts.every((p) => NAME_PART.test(p) && !/[ .]$/.test(p));
}

function normalizeEntry(entry) {
  return {
    lora: typeof entry?.lora === "string" ? entry.lora : "",
    strength: Number.isFinite(entry?.strength) ? entry.strength : 1,
    enabled: entry?.enabled !== false,
  };
}

function parseStack(text) {
  try {
    const data = JSON.parse(text);
    if (data?.schema_version === SCHEMA_VERSION && Array.isArray(data.entries)) {
      return data.entries.map(normalizeEntry);
    }
  } catch {
    // Fall through.
  }
  return null;
}

function serializeStack(entries) {
  return JSON.stringify({
    schema_version: SCHEMA_VERSION,
    entries: entries.map(({ lora, strength, enabled }) => ({ lora, strength, enabled })),
  });
}

const stackFile = (name) => `${DIR}/${name}.json`;

async function listStacks() {
  const files = await api.listUserDataFullInfo(DIR);
  return files
    .map((file) => file.path.replaceAll("\\", "/"))
    .filter((path) => path.endsWith(".json"))
    .map((path) => path.slice(0, -5))
    .filter(validName)
    .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
}

// Returns the saved entries, or null when no stack has that name.
async function readStack(name) {
  const response = await api.getUserData(stackFile(name));
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Could not load "${name}" (${response.status}).`);
  const entries = parseStack(await response.text());
  if (!entries) throw new Error(`"${name}" is not a valid LoRA stack file.`);
  return entries;
}

async function writeStack(name, entries) {
  await api.storeUserData(stackFile(name), serializeStack(entries), {
    overwrite: true, stringify: false, throwOnError: true, full_info: false,
  });
}

async function deleteStack(name) {
  const response = await api.deleteUserData(stackFile(name));
  if (!response.ok && response.status !== 404) throw new Error(`Could not delete "${name}" (${response.status}).`);
}

async function loadLoraList(refresh = false) {
  if (!loraList || refresh) {
    const response = await api.fetchApi("/models/loras");
    loraList = response.ok ? await response.json() : [];
  }
  return loraList;
}

// ---------------------------------------------------------------- dialogs

async function askText(title, message, defaultValue) {
  const dialog = app.extensionManager?.dialog;
  if (dialog?.prompt) return dialog.prompt({ title, message, defaultValue });
  return window.prompt(message, defaultValue);
}

async function confirmAction(title, message) {
  const dialog = app.extensionManager?.dialog;
  if (dialog?.confirm) return Boolean(await dialog.confirm({ title, message }));
  return window.confirm(message);
}

function notify(severity, summary, detail = "") {
  const toast = app.extensionManager?.toast;
  if (toast?.add) toast.add({ severity, summary, detail, life: severity === "error" ? 8000 : 3000 });
  else if (severity === "error") window.alert(`${summary}\n${detail}`);
}

// ---------------------------------------------------------------- library refresh

function graphNodes(graph = app.graph) {
  const nodes = [...(graph?._nodes ?? graph?.nodes ?? [])];
  for (const subgraph of graph?.subgraphs?.values?.() ?? []) nodes.push(...(subgraph._nodes ?? subgraph.nodes ?? []));
  return nodes;
}

async function refreshLibrary() {
  let names;
  try {
    names = await listStacks();
  } catch (error) {
    notify("error", "Could not list LoRA stacks", String(error.message ?? error));
    return;
  }
  for (const editor of editors) editor.setNames(names);
  for (const node of graphNodes()) {
    if (node.comfyClass !== BY_NAME_NODE) continue;
    const widget = node.widgets?.find((w) => w.name === "stack_name");
    if (widget) widget.options.values = names;
  }
  app.graph?.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------- editor

const STYLE = `
.moc-stack { display: flex; flex-direction: column; gap: 4px; font: 12px sans-serif; color: var(--input-text, #ddd); box-sizing: border-box; padding: 2px 0; }
.moc-stack select, .moc-stack input, .moc-stack button { font: inherit; color: inherit; background: var(--comfy-input-bg, #222); border: 1px solid var(--border-color, #444); border-radius: 4px; height: 22px; box-sizing: border-box; }
.moc-stack button { cursor: pointer; padding: 0 6px; }
.moc-stack button:hover:not(:disabled) { filter: brightness(1.3); }
.moc-stack button:disabled { opacity: .45; cursor: default; }
.moc-stack .bar { display: flex; gap: 4px; align-items: center; }
.moc-stack .bar select { flex: 1; min-width: 0; }
.moc-stack .status { white-space: nowrap; opacity: .8; }
.moc-stack .status.modified { color: var(--warning-text, #e0b030); opacity: 1; }
.moc-stack .status.missing { color: var(--error-text, #ff6b6b); opacity: 1; }
.moc-stack .rows { display: flex; flex-direction: column; gap: 2px; }
.moc-stack .row { display: flex; gap: 4px; align-items: center; border-top: 2px solid transparent; border-bottom: 2px solid transparent; }
.moc-stack .row.disabled > :not(.handle):not(.toggle):not(.remove) { opacity: .45; }
.moc-stack .row.drop-before { border-top-color: var(--p-primary-color, #4a9eff); }
.moc-stack .row.drop-after { border-bottom-color: var(--p-primary-color, #4a9eff); }
.moc-stack .handle { cursor: grab; user-select: none; padding: 0 2px; opacity: .7; }
.moc-stack .toggle { margin: 0; height: 14px; width: 14px; }
.moc-stack .lora { flex: 1; min-width: 0; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.moc-stack .lora.empty-choice { opacity: .6; }
.moc-stack .lora.missing { border-color: var(--error-text, #ff6b6b); color: var(--error-text, #ff6b6b); }
.moc-picker { position: fixed; z-index: 10000; display: flex; flex-direction: column; gap: 4px; padding: 4px; box-sizing: border-box; font: 12px sans-serif; color: var(--input-text, #ddd); background: var(--comfy-menu-bg, #2a2a2a); border: 1px solid var(--border-color, #444); border-radius: 6px; box-shadow: 0 4px 16px rgba(0, 0, 0, .5); }
.moc-picker input { font: inherit; color: inherit; background: var(--comfy-input-bg, #222); border: 1px solid var(--border-color, #444); border-radius: 4px; height: 24px; padding: 0 6px; box-sizing: border-box; outline: none; }
.moc-picker .list { overflow-y: auto; }
.moc-picker .item { padding: 3px 6px; border-radius: 3px; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.moc-picker .item .folder { opacity: .55; }
.moc-picker .item.current { font-weight: bold; }
.moc-picker .item.active { background: var(--p-primary-color, #4a9eff); color: #fff; }
.moc-picker .note { padding: 3px 6px; opacity: .6; }
.moc-stack .strength { width: 64px; padding: 0 4px; }
.moc-stack .empty { opacity: .6; padding: 4px 2px; }
`;

function ensureStyle() {
  if (document.getElementById("moc-lora-stack-style")) return;
  const style = document.createElement("style");
  style.id = "moc-lora-stack-style";
  style.textContent = STYLE;
  document.head.append(style);
}

function element(tag, props = {}, children = []) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

// ---------------------------------------------------------------- searchable LoRA picker

const PICKER_LIMIT = 300;
let closeOpenPicker = null;

// Every whitespace-separated term must appear somewhere in the path, in any order.
function matchesQuery(name, query) {
  const haystack = name.toLowerCase();
  return query.toLowerCase().split(/\s+/).filter(Boolean).every((term) => haystack.includes(term));
}

function pickerItemLabel(name) {
  const slash = name.lastIndexOf("/");
  if (slash < 0) return [name];
  return [element("span", { className: "folder", textContent: name.slice(0, slash + 1) }), name.slice(slash + 1)];
}

function openPicker(anchor, names, current, initialQuery, onPick) {
  closeOpenPicker?.();
  const search = element("input", { type: "text", placeholder: "Type to filter LoRAs…", value: initialQuery, spellcheck: false });
  const list = element("div", { className: "list" });
  const popup = element("div", { className: "moc-picker" }, [search, list]);
  let shown = [];
  let active = 0;

  const close = () => {
    popup.remove();
    document.removeEventListener("pointerdown", onOutside, true);
    window.removeEventListener("resize", close);
    if (closeOpenPicker === close) closeOpenPicker = null;
  };
  const onOutside = (event) => {
    if (!popup.contains(event.target)) close();
  };
  const pick = (name) => {
    close();
    anchor.focus();
    if (name !== undefined) onPick(name);
  };
  const highlight = (index) => {
    if (!shown.length) return;
    list.children[active]?.classList.remove("active");
    active = Math.max(0, Math.min(index, shown.length - 1));
    list.children[active]?.classList.add("active");
    list.children[active]?.scrollIntoView({ block: "nearest" });
  };
  const render = () => {
    const matches = names.filter((name) => matchesQuery(name, search.value));
    shown = matches.slice(0, PICKER_LIMIT);
    list.replaceChildren(...shown.map((name, index) => element("div", {
      className: name === current ? "item current" : "item",
      title: name,
      onpointerdown: (event) => {
        event.preventDefault();
        pick(name);
      },
      onpointermove: () => active !== index && highlight(index),
    }, pickerItemLabel(name))));
    if (!matches.length) list.append(element("div", { className: "note", textContent: names.length ? "No matching LoRAs" : "No LoRA files found" }));
    if (matches.length > shown.length) {
      list.append(element("div", { className: "note", textContent: `${matches.length - shown.length} more — keep typing to narrow down` }));
    }
    active = -1;
    const currentIndex = search.value ? -1 : shown.indexOf(current);
    highlight(currentIndex >= 0 ? currentIndex : 0);
  };

  search.addEventListener("input", render);
  search.addEventListener("keydown", (event) => {
    event.stopPropagation();
    const page = 10;
    if (event.key === "ArrowDown") highlight(active + 1);
    else if (event.key === "ArrowUp") highlight(active - 1);
    else if (event.key === "PageDown") highlight(active + page);
    else if (event.key === "PageUp") highlight(active - page);
    else if (event.key === "Enter") pick(shown[active]);
    else if (event.key === "Escape") pick(undefined);
    else if (event.key === "Tab") close();
    else return;
    event.preventDefault();
  });
  popup.addEventListener("wheel", (event) => event.stopPropagation());

  // Open below the field, or above it when there is more room there.
  const rect = anchor.getBoundingClientRect();
  const below = window.innerHeight - rect.bottom - 8;
  const above = rect.top - 8;
  const height = Math.min(360, Math.max(below, above));
  popup.style.left = `${Math.max(4, Math.min(rect.left, window.innerWidth - Math.max(rect.width, 320) - 4))}px`;
  popup.style.width = `${Math.max(rect.width, 320)}px`;
  popup.style.maxHeight = `${height}px`;
  list.style.maxHeight = `${height - 40}px`;
  if (below >= Math.min(360, above)) popup.style.top = `${rect.bottom + 2}px`;
  else popup.style.bottom = `${window.innerHeight - rect.top + 2}px`;

  document.body.append(popup);
  document.addEventListener("pointerdown", onOutside, true);
  window.addEventListener("resize", close);
  closeOpenPicker = close;
  render();
  search.focus();
  search.setSelectionRange(search.value.length, search.value.length);
}

class StackEditor {
  constructor(node) {
    this.node = node;
    this.entries = [];
    this.names = [];
    this.saved = undefined; // serialized saved stack; null when the linked file is missing
    this.dataWidget = node.widgets.find((w) => w.name === "stack_data");
    this.dataWidget.hidden = true;
    this.dataWidget.options = { ...this.dataWidget.options, hidden: true };
    this.dataWidget.computeSize = () => [0, -4];

    ensureStyle();
    this.nameSelect = element("select", { title: "Saved stacks", onchange: () => this.selectStack(this.nameSelect.value) });
    this.status = element("span", { className: "status" });
    const button = (text, title, onclick) => element("button", { textContent: text, title, onclick });
    this.saveButton = button("Save", "Save the rows to the linked stack", () => this.save());
    this.revertButton = button("Revert", "Discard changes and reload the linked stack", () => this.revert());
    this.deleteButton = button("Delete", "Delete the linked saved stack", () => this.remove());
    this.rows = element("div", { className: "rows" });
    this.root = element("div", { className: "moc-stack" }, [
      element("div", { className: "bar" }, [
        this.nameSelect,
        button("↻", "Refresh saved stacks and LoRA files", () => this.refresh()),
      ]),
      element("div", { className: "bar" }, [
        this.saveButton,
        button("Save as…", "Save the rows under a new name", () => this.saveAs()),
        this.revertButton,
        this.deleteButton,
        this.status,
      ]),
      this.rows,
      element("div", { className: "bar" }, [button("+ Add LoRA", "Add a LoRA at the bottom", () => this.addRow())]),
    ]);
    // Keep pointer, wheel, and key input inside the editor instead of panning the canvas or
    // triggering shortcuts (Delete would otherwise remove the node while a row is focused).
    for (const type of ["pointerdown", "wheel", "keydown"]) this.root.addEventListener(type, (event) => event.stopPropagation());

    this.widget = node.addDOMWidget("stack_editor", "moc_lora_stack", this.root, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => this.height(),
      getMaxHeight: () => this.height(),
    });
    this.loadFromWidget();
    loadLoraList().then(() => this.renderRows()).catch(() => {});
    listStacks().then((names) => this.setNames(names)).catch(() => {});
  }

  get linked() {
    return this.node.properties?.[LINK_PROPERTY] ?? "";
  }

  set linked(name) {
    this.node.properties ??= {};
    this.node.properties[LINK_PROPERTY] = name;
  }

  height() {
    return 3 * 22 + 4 * 4 + Math.max(1, this.entries.length) * 28 + 4;
  }

  isModified() {
    return typeof this.saved === "string" && serializeStack(this.entries) !== this.saved;
  }

  // ---- state

  loadFromWidget() {
    this.entries = parseStack(this.dataWidget.value) ?? [];
    this.saved = undefined;
    this.renderRows();
    this.updateBar();
    const name = this.linked;
    if (name) {
      readStack(name)
        .then((entries) => {
          if (this.linked === name) this.saved = entries ? serializeStack(entries) : null;
        })
        .catch(() => {})
        .finally(() => this.updateBar());
    }
  }

  commit() {
    const value = serializeStack(this.entries);
    this.dataWidget.value = value;
    this.dataWidget.callback?.(value);
    this.updateBar();
    this.node.graph?.setDirtyCanvas?.(true, true);
  }

  structureChanged() {
    this.commit();
    this.renderRows();
    const size = this.node.computeSize();
    this.node.setSize([Math.max(this.node.size[0], size[0]), size[1]]);
  }

  setNames(names) {
    this.names = names;
    this.updateBar();
  }

  // ---- library actions

  async selectStack(name) {
    if (name === this.linked) return;
    if (!name) {
      this.linked = "";
      this.saved = undefined;
      this.updateBar();
      return;
    }
    if (this.isModified() || (!this.linked && this.entries.length)) {
      const ok = await confirmAction("Load LoRA stack", `Replace the current rows with "${name}"?`);
      if (!ok) return this.updateBar();
    }
    await this.loadNamed(name);
  }

  async loadNamed(name) {
    try {
      const entries = await readStack(name);
      if (!entries) {
        notify("error", "LoRA stack not found", `"${name}" no longer exists.`);
        return refreshLibrary();
      }
      this.entries = entries;
      this.linked = name;
      this.saved = serializeStack(entries);
      this.structureChanged();
    } catch (error) {
      notify("error", "Could not load LoRA stack", String(error.message ?? error));
      this.updateBar();
    }
  }

  async save() {
    if (!this.linked) return this.saveAs();
    await this.store(this.linked);
  }

  async saveAs() {
    let name = await askText("Save LoRA stack", "Stack name (use / for subfolders):", this.linked || "");
    name = (name ?? "").trim().replace(/\.json$/i, "");
    if (!name) return;
    if (!validName(name)) {
      return notify("error", "Invalid stack name", 'Use letters, digits, spaces, "_", "-", "." and "/" for subfolders.');
    }
    if (name !== this.linked && this.names.includes(name)) {
      if (!(await confirmAction("Overwrite LoRA stack", `"${name}" already exists. Overwrite it?`))) return;
    }
    await this.store(name);
  }

  async store(name) {
    try {
      await writeStack(name, this.entries);
      this.linked = name;
      this.saved = serializeStack(this.entries);
      notify("success", "LoRA stack saved", name);
    } catch (error) {
      notify("error", "Could not save LoRA stack", String(error.message ?? error));
    }
    await refreshLibrary();
  }

  async revert() {
    if (this.linked) await this.loadNamed(this.linked);
  }

  async remove() {
    const name = this.linked;
    if (!name || !(await confirmAction("Delete LoRA stack", `Delete the saved stack "${name}"? The rows on this node stay.`))) return;
    try {
      await deleteStack(name);
      this.linked = "";
      this.saved = undefined;
      notify("success", "LoRA stack deleted", name);
    } catch (error) {
      notify("error", "Could not delete LoRA stack", String(error.message ?? error));
    }
    await refreshLibrary();
  }

  async refresh() {
    await loadLoraList(true).catch(() => {});
    this.renderRows();
    await refreshLibrary();
    if (this.linked) {
      const name = this.linked;
      const entries = await readStack(name).catch(() => undefined);
      if (entries !== undefined && this.linked === name) this.saved = entries ? serializeStack(entries) : null;
      this.updateBar();
    }
  }

  // ---- rendering

  updateBar() {
    const linked = this.linked;
    const options = [element("option", { value: "", textContent: "— not saved —" })];
    const names = linked && !this.names.includes(linked) ? [...this.names, linked] : this.names;
    for (const name of names) options.push(element("option", { value: name, textContent: name }));
    this.nameSelect.replaceChildren(...options);
    this.nameSelect.value = linked;

    let text = "", className = "status";
    if (!linked) text = this.entries.length ? "unsaved" : "";
    else if (this.saved === null) [text, className] = ["saved file missing", "status missing"];
    else if (this.isModified()) [text, className] = ["● modified", "status modified"];
    else if (typeof this.saved === "string") text = "saved";
    this.status.textContent = text;
    this.status.className = className;
    this.saveButton.disabled = Boolean(linked) && typeof this.saved === "string" && !this.isModified();
    this.revertButton.disabled = !linked || !this.isModified();
    this.deleteButton.disabled = !linked || this.saved === null;
  }

  addRow() {
    this.entries.push({ lora: "", strength: 1, enabled: true });
    this.structureChanged();
  }

  moveRow(from, to) {
    if (from === to || from + 1 === to) return;
    const [entry] = this.entries.splice(from, 1);
    this.entries.splice(to > from ? to - 1 : to, 0, entry);
    this.structureChanged();
  }

  renderRows() {
    closeOpenPicker?.();
    if (!this.entries.length) {
      this.rows.replaceChildren(element("div", { className: "empty", textContent: "No LoRAs. The model passes through unchanged." }));
      return;
    }
    this.rows.replaceChildren(...this.entries.map((entry, index) => this.renderRow(entry, index)));
  }

  renderRow(entry, index) {
    const available = loraList ?? [];
    const missing = entry.lora && loraList !== null && !available.includes(entry.lora);
    const lora = element("button", {
      className: `lora${missing ? " missing" : ""}${entry.lora ? "" : " empty-choice"}`,
      textContent: entry.lora ? `${entry.lora}${missing ? " (missing)" : ""}` : "Choose LoRA…",
      title: missing ? `${entry.lora} is not in the loras folders` : entry.lora || "Click or start typing to search LoRAs",
    });
    const choose = (query = "") => openPicker(lora, available, entry.lora, query, (name) => {
      if (name === entry.lora) return;
      entry.lora = name;
      this.structureChanged();
      this.rows.children[index]?.querySelector(".lora")?.focus();
    });
    lora.addEventListener("click", () => choose());
    lora.addEventListener("keydown", (event) => {
      const typed = event.key.length === 1 && event.key !== " " && !event.ctrlKey && !event.metaKey && !event.altKey;
      if (!typed && !["Enter", " ", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      event.stopPropagation();
      choose(typed ? event.key : "");
    });

    const strength = element("input", {
      className: "strength", type: "number", step: 0.05, min: -STRENGTH_LIMIT, max: STRENGTH_LIMIT,
      value: String(entry.strength), title: "Strength",
      onchange: () => {
        const value = Number(strength.value);
        entry.strength = Number.isFinite(value) ? Math.max(-STRENGTH_LIMIT, Math.min(STRENGTH_LIMIT, value)) : 1;
        strength.value = String(entry.strength);
        this.commit();
      },
    });
    const row = element("div", { className: entry.enabled ? "row" : "row disabled" }, [
      element("span", { className: "handle", textContent: "≡", title: "Drag to reorder" }),
      element("input", {
        className: "toggle", type: "checkbox", checked: entry.enabled, title: "Enabled",
        onchange: (event) => {
          entry.enabled = event.target.checked;
          row.classList.toggle("disabled", !entry.enabled);
          this.commit();
        },
      }),
      lora,
      strength,
      element("button", {
        className: "remove", textContent: "✕", title: "Remove",
        onclick: () => {
          this.entries.splice(index, 1);
          this.structureChanged();
        },
      }),
    ]);

    // Reorder by dragging the handle; dropping on a row's upper or lower half inserts above or below it.
    const handle = row.firstChild;
    const below = (event) => event.clientY > row.getBoundingClientRect().top + row.offsetHeight / 2;
    const clearTarget = () => row.classList.remove("drop-before", "drop-after");
    handle.addEventListener("pointerdown", () => (row.draggable = true));
    row.addEventListener("dragstart", (event) => {
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/x-moc-lora-row", String(index));
    });
    row.addEventListener("dragend", () => (row.draggable = false));
    row.addEventListener("dragover", (event) => {
      if (!event.dataTransfer.types.includes("text/x-moc-lora-row")) return;
      event.preventDefault();
      clearTarget();
      row.classList.add(below(event) ? "drop-after" : "drop-before");
    });
    row.addEventListener("dragleave", clearTarget);
    row.addEventListener("drop", (event) => {
      event.preventDefault();
      clearTarget();
      const from = Number(event.dataTransfer.getData("text/x-moc-lora-row"));
      if (Number.isInteger(from)) this.moveRow(from, below(event) ? index + 1 : index);
    });
    return row;
  }
}

app.registerExtension({
  name: "moc.LoraStack",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== STACK_NODE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.mocStackEditor = new StackEditor(this);
      editors.add(this.mocStackEditor);
      this.setSize([Math.max(this.size[0], 380), this.computeSize()[1]]);
      return result;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.mocStackEditor?.loadFromWidget();
      return result;
    };
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      closeOpenPicker?.();
      editors.delete(this.mocStackEditor);
      return onRemoved?.apply(this, arguments);
    };
  },
});
