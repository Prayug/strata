/* AI Lab entry: the landing page, or one research notebook. */
import { useParams } from "react-router-dom";
import { Landing } from "./Landing";
import { Workspace } from "./Workspace";

export default function Lab() {
  const { sid } = useParams();
  return sid ? <Workspace key={sid} sid={sid} /> : <Landing />;
}
