"use strict";

const assert = require("node:assert/strict");
const { test } = require("node:test");
const { alignTexts, baselineEvaluation, evaluationStep } = require("../../reporting/berrutti/web/app.js");

test("progress controls precede the final validation of their actual epoch", () => {
  const epochs = [208, 2, 104, 1, 204, 112, 108, 0];
  assert.deepEqual(epochs.sort((a, b) => evaluationStep({ epoch: a }) - evaluationStep({ epoch: b })), [0, 104, 108, 112, 1, 204, 208, 2]);
});

function restored(result, id) {
  return result.rows.flatMap((row) => row.cells[id]).map((run) => run.text).join("");
}

test("all three anchors preserve every character through insertions and deletions", () => {
  const texts = { reference: "  Uno\n dos tres cuatro.  ", base: "Uno dos extra cuatro.\n", tuned: "\tUno dos tres cuatro.\nfin" };
  for (const anchor of Object.keys(texts)) {
    const result = alignTexts(texts, anchor);
    for (const id of Object.keys(texts)) assert.equal(restored(result, id), texts[id]);
    assert(result.rows.some((row) => row.changed));
    assert(result.rows.flatMap((row) => row.cells[anchor]).some((run) => run.kind === "del"));
  }
});

test("base-to-tuned substitutions align without a human reference", () => {
  const result = alignTexts({ base: "a incorrecto c", tuned: "a correcto c" }, "base");
  assert.deepEqual(result.rows[0].cells.base.map((run) => run.kind), ["same", "del", "same"]);
  assert.deepEqual(result.rows[0].cells.tuned.map((run) => run.kind), ["same", "ins", "same"]);
  assert.equal(restored(result, "base"), "a incorrecto c");
  assert.equal(restored(result, "tuned"), "a correcto c");
});

test("one-panel pairwise runs reconstruct both texts in either direction", () => {
  const examples = [
    { reference: "  Uno\n dos tres cuatro.  ", base: "Uno dos extra cuatro.\n", tuned: "\tUno dos tres cuatro.\nfin" },
    { reference: "", base: "insertado", tuned: "" },
    { reference: "a a b", base: "a b b", tuned: "a\n a b" },
    { reference: "<img src=x onerror=alert(1)>", base: "<script>test</script>", tuned: "& é\n" },
    { reference: "uno ".repeat(70), base: "dos ".repeat(50), tuned: "uno ".repeat(85) },
  ];
  for (const texts of examples) {
    for (const anchor of Object.keys(texts)) {
      for (const other of Object.keys(texts).filter((id) => id !== anchor)) {
        const result = alignTexts({ [anchor]: texts[anchor], [other]: texts[other] }, anchor);
        const runs = result.rows.flatMap((row) => row.runs);
        assert.equal(runs.filter((run) => run.kind !== "ins").map((run) => run.text).join(""), texts[anchor]);
        assert.equal(runs.filter((run) => run.kind !== "del").map((run) => run.text).join(""), texts[other]);
        assert(runs.every((run) => ["same", "del", "ins"].includes(run.kind)));
      }
    }
  }
});

test("empty predictions, repeated words, whitespace and markup remain literal", () => {
  for (const texts of [
    { reference: "", base: "insertado", tuned: "" },
    { reference: "una una dos", base: "una dos dos", tuned: "una una dos" },
    { reference: "a b", base: "a\n b", tuned: "a b" },
    { reference: "<img src=x onerror=alert(1)> & é", base: "", tuned: "<script>test</script>" },
  ]) {
    for (const anchor of Object.keys(texts)) {
      const result = alignTexts(texts, anchor);
      for (const id of Object.keys(texts)) assert.equal(restored(result, id), texts[id]);
    }
  }
  assert(alignTexts({ reference: "igual", base: "igual", tuned: "igual" }, "reference").rows.every((row) => !row.changed));
  assert.throws(() => alignTexts({ reference: "a", base: null }, "base"));
});

test("large and highly fragmented texts fall back within the table bound", () => {
  for (const separator of [" ", "\n"]) {
    const source = Array.from({ length: 1500 }, (_, index) => `palabra${index}`).join(separator);
    const result = alignTexts({ reference: source, base: source + " agregado", tuned: source }, "reference");
    assert.equal(result.granularity, separator === " " ? "líneas" : "bloques completos");
    for (const id of ["reference", "base", "tuned"]) assert.equal(restored(result, id), id === "base" ? source + " agregado" : source);
    const pair = alignTexts({ reference: source, base: source + " agregado" }, "reference");
    const runs = pair.rows.flatMap((row) => row.runs);
    assert.equal(runs.filter((run) => run.kind !== "ins").map((run) => run.text).join(""), source);
    assert.equal(runs.filter((run) => run.kind !== "del").map((run) => run.text).join(""), source + " agregado");
  }
});

test("baseline lookup stays within the split and uses only the extension's own baseline", () => {
  const validation = { method: "baseline", stage: "baseline", epoch: 0, split: "validation" };
  const heldout = { ...validation, split: "test" };
  const own = { ...validation, method: "experiment4_baseline" };
  const catalog = { evaluations: [validation, heldout, own, { method: "pages", epoch: 1, split: "validation" }] };
  for (const method of ["pages", "lines", "combined"]) {
    assert.equal(baselineEvaluation(catalog, method, "validation"), validation);
    assert.equal(baselineEvaluation(catalog, method, "test"), heldout);
  }
  assert.equal(baselineEvaluation(catalog, "experiment4", "validation"), own);
  assert.equal(baselineEvaluation(catalog, "experiment4", "test"), null);
  assert.equal(baselineEvaluation(catalog, "pages", "train"), null);
  assert.throws(() => baselineEvaluation({ evaluations: [validation, { ...validation }] }, "pages", "validation"));
});
