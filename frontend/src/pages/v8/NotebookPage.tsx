/* Route wrapper: one notebook per session id (a new id is a fresh component). */
import { useParams } from "react-router-dom";
import { Notebook } from "./Notebook";

export default function NotebookPage() {
  const { sid } = useParams();
  return sid ? <Notebook key={sid} sid={sid} /> : null;
}
