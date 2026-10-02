// paste this into console in js unless you like pain. in which case copy and paste data_schema.txt into a folder.
async function insert(folderName, autoreplace, txt) { // suggestion: use String.raw`...`
    const sleep = (ms) => new Promise(_ => setTimeout(_, ms));
    const state = Calc.getState();
    const folderId = `notes_from_${Date.now()}`;

  const newFolder = {
    id: folderId,
    type: "folder",
    title: folderName,
    collapsed: true,
    hidden: true
  };

  const lines = txt.split(/\r?\n/).filter(line => line.length > 0).map((line, i) => ({
    id: `${folderId}_line_${i}`,
    latex: autoreplace ? line.replace("[", "\\left[").replace("]", "\\right]").replace("(", "\\left(").replace(")", "\\right)") : line
  }));

  txt = null;

  const dummyLines = lines.map((_, i) => ({
    id: `${folderId}_line_${i}`,
    type: "expression",
    latex: "",
    folderId: folderId
  }));

  state.expressions.list.push(newFolder, ...dummyLines);
  state.expressions.list.push({
    id: `${folderId}_temp_stats`,
    type: "expression",
    latex: `${dummyLines.length}`,
  });
  Calc.setState(state);

  for (const [index, line] of lines.entries()) {
    Calc.setExpression({
      id: `${folderId}_temp_stats`,
      type: "expression",
      latex: `${dummyLines.length - (index + 1)}`,
    });
    Calc.setExpression(line);
    await sleep(1e-3 * line.latex.length); // For very large graphs recommended to increase 1e-3 to 2e-3
    lines[index] = null;
  }

  Calc.removeExpression({ id: `${folderId}_temp_stats` });
}
