import React from 'react';
import styles from './Tappable.module.css';

const Tappable = ({ children, onClick, style }) => {
  return (
    <div className={styles.main} style={style} onClick={onClick}>
      {children}
    </div>
  );
};

export default Tappable;
