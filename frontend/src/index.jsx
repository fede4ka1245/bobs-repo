import React from 'react';

import ReactDOM from 'react-dom/client';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <iframe
    src={'/streamlitqa'}
    className={'frame'}
  />
);
