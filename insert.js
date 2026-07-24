// paste this into console in js unless you like pain. in which case copy and paste data_schema.txt into a folder.
function insert(txt) { // suggestion: use String.raw`...`
  const state = Calc.getState();
  const folderId = `notes_from_${Date.now()}`;

  const newFolder = {
    id: folderId,
    type: "folder",
    title: `Shard_${Date.now()}`,
    collapsed: true,
    hidden: true
  };

  const lines = txt.split(/\r?\n/).map((line, i) => ({
    id: `line_${i}`,
    type: "expression",
    latex: line,
    folderId: folderId
  }));

  state.expressions.list.push(newFolder, ...lines)
  Calc.setState(state);
}
